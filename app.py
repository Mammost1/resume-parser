import streamlit as st
import cv2
import numpy as np
import json
import re
import base64
import requests
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from pythainlp.util import normalize as th_normalize
from sentence_transformers import SentenceTransformer, util

# ==================== CONFIG ====================
VISION_API_KEY = 'AIzaSyC484F0-HAfCTvbScezwUntHy5efpJCRRA'
VISION_URL = f'https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}'
CLASS_NAMES = ['Personality', 'Education', 'Experience', 'Skill', 'Project', 'Training']
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.15
CROP_PAD_RATIO = 0.03
MIN_CROP_HEIGHT = 50

COLORS = {
    'Personality': '#FF6B6B',
    'Education':   '#4ECDC4',
    'Experience':  '#45B7D1',
    'Skill':       '#96CEB4',
    'Project':     '#FFEAA7',
    'Training':    '#DDA0DD',
}

# ==================== Load Models ====================
@st.cache_resource
def load_models():
    yolo = YOLO('best_model.pt')
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df_onet = pd.read_excel('Skills.xlsx')
    onet_skills = sorted(df_onet['Element Name'].unique().tolist())
    onet_occs = sorted(df_onet['Title'].unique().tolist())
    skill_emb = embedder.encode(onet_skills, convert_to_tensor=True, show_progress_bar=False)
    occ_emb   = embedder.encode(onet_occs,   convert_to_tensor=True, show_progress_bar=False)
    return yolo, embedder, df_onet, onet_skills, onet_occs, skill_emb, occ_emb

# ==================== Helper Functions ====================
def preprocess_crop(crop_img):
    h, w = crop_img.shape[:2]
    if h < MIN_CROP_HEIGHT:
        scale = MIN_CROP_HEIGHT / h
        crop_img = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop_img

def postprocess_thai(text):
    if not text:
        return text
    text = th_normalize(text)
    text = ' '.join(text.split())
    return text

def extract_text_from_crop(crop_img):
    processed = preprocess_crop(crop_img)
    _, buffer = cv2.imencode('.jpg', processed)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    payload = {
        'requests': [{
            'image': {'content': img_base64},
            'features': [{'type': 'TEXT_DETECTION'}],
            'imageContext': {'languageHints': ['th', 'en']}
        }]
    }
    try:
        response = requests.post(VISION_URL, json=payload, timeout=10)
        result = response.json()
        raw_text = ''
        if 'responses' in result and result['responses']:
            annotations = result['responses'][0].get('textAnnotations', [])
            if annotations:
                raw_text = annotations[0].get('description', '').strip()
    except Exception:
        raw_text = ''
    clean_text = postprocess_thai(raw_text)
    return raw_text, clean_text

def parse_resume(image_path, yolo_model):
    results = yolo_model.predict(str(image_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    sections = []
    for r in results:
        img_bgr = r.orig_img
        h, w = img_bgr.shape[:2]
        boxes = sorted(r.boxes, key=lambda x: x.xyxy[0][1])
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = yolo_model.names[cls_id]
            pad_x  = int((x2 - x1) * CROP_PAD_RATIO)
            pad_y  = int((y2 - y1) * CROP_PAD_RATIO)
            cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            raw_text, clean_text = extract_text_from_crop(crop)
            sections.append({
                'type': label, 'confidence': round(conf, 4),
                'bbox': [x1, y1, x2, y2],
                'raw_text': raw_text, 'clean_text': clean_text,
            })
    return {'sections': sections}

def split_lines(text):
    if not text:
        return []
    out = []
    for line in text.split('\n'):
        for p in re.split(r'[,•·|/;]+|\s{2,}', line):
            p = p.strip(' \t-*.•&:')
            if 2 <= len(p) <= 100:
                out.append(p)
    return out

def best_match(query, items, embeddings, embedder, threshold):
    emb = embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)
    cos = util.cos_sim(emb, embeddings)[0]
    top = torch.topk(cos, k=1)
    score, idx = top.values[0].item(), top.indices[0].item()
    if score >= threshold:
        return {'name': items[idx], 'similarity': round(score, 4)}
    return None

def get_skill_value(df_onet, onet_skill, scale='IM'):
    rows = df_onet[
        (df_onet['Element Name'] == onet_skill) & (df_onet['Scale ID'] == scale)
    ]['Data Value']
    if rows.empty:
        return {}
    return {'avg_value': round(rows.mean(), 2), 'max_value': round(rows.max(), 2)}

def map_resume(parsed_result, embedder, df_onet, onet_skills, onet_occs, skill_emb, occ_emb, skill_th=0.40, occ_th=0.45):
    skill_text = '\n'.join(s.get('raw_text', '') for s in parsed_result['sections'] if s['type'] == 'Skill')
    skill_seen = {}
    for chunk in split_lines(skill_text):
        m = best_match(chunk, onet_skills, skill_emb, embedder, skill_th)
        if m and (m['name'] not in skill_seen or m['similarity'] > skill_seen[m['name']]['similarity']):
            val = get_skill_value(df_onet, m['name'])
            skill_seen[m['name']] = {'resume_text': chunk, **m, **val}

    exp_text = '\n'.join(s.get('raw_text', '') for s in parsed_result['sections'] if s['type'] == 'Experience')
    occ_seen = {}
    for chunk in split_lines(exp_text):
        m = best_match(chunk, onet_occs, occ_emb, embedder, occ_th)
        if m and (m['name'] not in occ_seen or m['similarity'] > occ_seen[m['name']]['similarity']):
            occ_seen[m['name']] = {'resume_text': chunk, **m}

    return {
        'skills':      sorted(skill_seen.values(), key=lambda x: -x['similarity']),
        'occupations': sorted(occ_seen.values(),   key=lambda x: -x['similarity']),
    }

def calculate_fit_score(candidate_skills, target_occupation, df_onet, scale='IM'):
    occ_reqs = df_onet[
        (df_onet['Title'] == target_occupation) & (df_onet['Scale ID'] == scale)
    ][['Element Name', 'Data Value']]
    if occ_reqs.empty:
        return {}
    candidate_skill_names = {s['name'] for s in candidate_skills}
    matched_score = 0; total_possible = 0; breakdown = []
    for _, row in occ_reqs.iterrows():
        skill = row['Element Name']; importance = float(row['Data Value'])
        total_possible += importance
        has_it = skill in candidate_skill_names
        if has_it:
            matched_score += importance
        breakdown.append({'skill': skill, 'importance': round(importance, 2), 'candidate_has': has_it})
    fit_pct = (matched_score / total_possible * 100) if total_possible else 0
    breakdown.sort(key=lambda x: -x['importance'])
    return {
        'fit_percentage': round(fit_pct, 1),
        'has_skills':     [b for b in breakdown if b['candidate_has']],
        'missing_skills': [b for b in breakdown if not b['candidate_has']],
    }

def visualize_boxes(img_bgr, sections):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img_rgb)
    for s in sections:
        x1, y1, x2, y2 = s['bbox']
        color = COLORS.get(s['type'], '#FFFFFF')
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        ))
        ax.text(x1, y1 - 8, f"{s['type']} {s['confidence']:.2f}",
                fontsize=10, color=color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# ==================== Streamlit UI ====================
st.set_page_config(page_title='Resume Parser v2', layout='wide', page_icon='📄')
st.title('📄 Resume Parser v2 — AI Skill Analyzer')
st.caption('Upload resume → ตรวจจับ section → จับคู่กับ O*NET → แนะนำอาชีพ')

with st.spinner('กำลังโหลด model...'):
    yolo_model, embedder, df_onet, onet_skills, onet_occs, skill_emb, occ_emb = load_models()

# Sidebar
with st.sidebar:
    st.header('⚙️ ตั้งค่า')
    skill_th = st.slider('Skill Threshold',   0.30, 0.70, 0.40, 0.05)
    occ_th   = st.slider('Occupation Threshold', 0.30, 0.70, 0.45, 0.05)
    top_n    = st.slider('Top N อาชีพแนะนำ', 5, 20, 10)
    target_occ = st.selectbox('🎯 อาชีพเป้าหมาย (optional)', [''] + onet_occs)

# Upload
uploaded_file = st.file_uploader('📤 Upload Resume (JPG / PNG)', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert('RGB')
    tmp_path = '/tmp/uploaded_resume.jpg'
    img_pil.save(tmp_path)

    with st.spinner('กำลังวิเคราะห์... อาจใช้เวลา 20-60 วินาที'):
        parsed   = parse_resume(tmp_path, yolo_model)
        mapping  = map_resume(parsed, embedder, df_onet, onet_skills, onet_occs,
                              skill_emb, occ_emb, skill_th, occ_th)
        img_bgr  = cv2.imread(tmp_path)
        annotated = visualize_boxes(img_bgr, parsed['sections'])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('📍 ตำแหน่ง Section')
        st.image(annotated, use_column_width=True)
    with col2:
        st.subheader('📋 สรุปผล')
        st.metric('Section ที่พบ', len(parsed['sections']))
        st.metric('Skills ที่ match', len(mapping['skills']))
        st.metric('Occupations ที่ match', len(mapping['occupations']))

        if target_occ:
            fit = calculate_fit_score(mapping['skills'], target_occ, df_onet)
            if fit:
                st.metric(f'Fit Score: {target_occ[:30]}', f"{fit['fit_percentage']}%")
                with st.expander('ดูรายละเอียด Fit Score'):
                    st.write(f"✅ Skills ที่มี: {len(fit['has_skills'])}")
                    for s in fit['has_skills'][:5]:
                        st.write(f"  • {s['skill']} (Importance {s['importance']})")
                    st.write(f"❌ Skills ที่ขาด: {len(fit['missing_skills'])}")
                    for s in fit['missing_skills'][:5]:
                        st.write(f"  • {s['skill']} (Importance {s['importance']})")

    tab1, tab2, tab3, tab4 = st.tabs(['🎯 Skills', '💼 Occupations', '🏆 Top Jobs', '📝 Raw JSON'])

    with tab1:
        if mapping['skills']:
            df_skills = pd.DataFrame([{
                'O*NET Skill':    s['name'],
                'Similarity':     s['similarity'],
                'Avg Importance': s.get('avg_value', '-'),
                'Resume Text':    s['resume_text'][:60],
            } for s in mapping['skills']])
            st.dataframe(df_skills, use_container_width=True)
        else:
            st.info('ไม่พบ Skills ที่ match')

    with tab2:
        if mapping['occupations']:
            df_occs = pd.DataFrame([{
                'Occupation':  o['name'],
                'Similarity':  o['similarity'],
                'Resume Text': o['resume_text'][:60],
            } for o in mapping['occupations']])
            st.dataframe(df_occs, use_container_width=True)
        else:
            st.info('ไม่พบ Occupations ที่ match')

    with tab3:
        with st.spinner('กำลังคำนวณ Top Jobs...'):
            top_jobs = []
            for occ in onet_occs[:200]:
                score = calculate_fit_score(mapping['skills'], occ, df_onet)
                if score:
                    top_jobs.append({'Occupation': occ, 'Fit %': score['fit_percentage'],
                                     'Has': len(score['has_skills']),
                                     'Missing': len(score['missing_skills'])})
            top_jobs = sorted(top_jobs, key=lambda x: -x['Fit %'])[:top_n]
            st.dataframe(pd.DataFrame(top_jobs), use_container_width=True)

    with tab4:
        st.json(parsed)
