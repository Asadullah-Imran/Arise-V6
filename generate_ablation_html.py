#!/usr/bin/env python3
"""
ARISE-V6 Ablation Visualizer Generator
Generates comprehensive interactive dashboards structured around the 5 evaluation categories:
1. 🎯 DEC Best Silhouette (Selected)
2. 🏆 DEC Best ARI Observed
3. 🏁 DEC Last Epoch (Terminal)
4. 🌱 Pre-train Best Silhouette
5. 🌿 Pre-train Best ARI
"""

import os
import re
import json
import numpy as np

def parse_ablation_log(filepath, track_name):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return [], []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    blocks = re.split(r'SEED SUMMARY:\s*', text)
    seed_entries = []
    
    for b in blocks[1:]:
        header_line = b.split('\n')[0]
        parts = [p.strip() for p in header_line.split('|')]
        variant = parts[0].replace('│', '').strip()
        seed_m = re.search(r'Seed:\s*(\d+)', header_line)
        ds_m = re.search(r'Dataset:\s*([\w\-]+)', header_line)
        if not seed_m or not ds_m:
            continue
        seed = int(seed_m.group(1))
        dataset = ds_m.group(1)
        
        def extract_row_nums(label):
            for line in b.split('\n'):
                if label in line:
                    cols = [c.strip() for c in line.split('│')]
                    cols = [c for c in cols if c]
                    if len(cols) >= 5:
                        phase = cols[1]
                        epoch = cols[2]
                        try:
                            sil = float(cols[3])
                            ari = float(cols[4])
                            return phase, epoch, sil, ari
                        except ValueError:
                            pass
            return '-', '-', 0.0, 0.0

        p1, e1, dec_best_sil, dec_best_sil_corr_ari = extract_row_nums('DEC Best Silhouette')
        p2, e2, dec_best_ari_corr_sil, dec_best_ari = extract_row_nums('DEC Best ARI Observed')
        p3, e3, dec_last_sil, dec_last_ari = extract_row_nums('DEC Last Epoch')
        p4, e4, pre_best_sil, pre_best_sil_corr_ari = extract_row_nums('Pre-train Best Silhouette')
        p5, e5, pre_best_ari_corr_sil, pre_best_ari = extract_row_nums('Pre-train Best ARI')
        
        rt_m = re.search(r'Total Training Runtime:\s*([\d\.]+)s', b)
        rt = float(rt_m.group(1)) if rt_m else 0.0
        
        # Structure the 5 explicit categories
        categories = {
            'dec_best_sil': {
                'name': 'DEC Best Silhouette (Selected)',
                'icon': '🎯',
                'phase': p1,
                'epoch': e1,
                'sil': dec_best_sil,
                'ari': dec_best_sil_corr_ari,
                'primary_val': dec_best_sil_corr_ari,
                'secondary_val': dec_best_sil
            },
            'dec_best_ari': {
                'name': 'DEC Best ARI Observed',
                'icon': '🏆',
                'phase': p2,
                'epoch': e2,
                'sil': dec_best_ari_corr_sil,
                'ari': dec_best_ari,
                'primary_val': dec_best_ari,
                'secondary_val': dec_best_ari_corr_sil
            },
            'dec_last_epoch': {
                'name': 'DEC Last Epoch (Terminal)',
                'icon': '🏁',
                'phase': p3,
                'epoch': e3,
                'sil': dec_last_sil,
                'ari': dec_last_ari,
                'primary_val': dec_last_ari,
                'secondary_val': dec_last_sil
            },
            'pretrain_best_sil': {
                'name': 'Pre-train Best Silhouette',
                'icon': '🌱',
                'phase': p4,
                'epoch': e4,
                'sil': pre_best_sil,
                'ari': pre_best_sil_corr_ari,
                'primary_val': pre_best_sil_corr_ari,
                'secondary_val': pre_best_sil
            },
            'pretrain_best_ari': {
                'name': 'Pre-train Best ARI',
                'icon': '🌿',
                'phase': p5,
                'epoch': e5,
                'sil': pre_best_ari_corr_sil,
                'ari': pre_best_ari,
                'primary_val': pre_best_ari,
                'secondary_val': pre_best_ari_corr_sil
            }
        }
        
        seed_entries.append({
            'track': track_name,
            'dataset': dataset,
            'variant': variant,
            'seed': seed,
            'categories': categories,
            'best_dec_sil': dec_best_sil,
            'best_dec_sil_corr_ari': dec_best_sil_corr_ari,
            'best_dec_sil_epoch': e1,
            'best_dec_ari': dec_best_ari,
            'best_dec_ari_corr_sil': dec_best_ari_corr_sil,
            'best_dec_ari_epoch': e2,
            'last_epoch_ari': dec_last_ari,
            'last_epoch_sil': dec_last_sil,
            'pretrain_best_sil': pre_best_sil,
            'pretrain_best_sil_corr_ari': pre_best_sil_corr_ari,
            'pretrain_best_sil_epoch': e4,
            'pretrain_best_ari': pre_best_ari,
            'pretrain_best_ari_corr_sil': pre_best_ari_corr_sil,
            'pretrain_best_ari_epoch': e5,
            'train_time_sec': rt
        })

    datasets = re.split(r'DATASET:\s*([\w\-]+)', text)
    epoch_records = []
    
    for i in range(1, len(datasets), 2):
        ds_name = datasets[i]
        ds_text = datasets[i+1]
        variants = re.split(r'VARIANT:\s*([\w\-]+)\s+on\s+[\w\-]+', ds_text)
        for j in range(1, len(variants), 2):
            var_name = variants[j]
            var_text = variants[j+1]
            pattern = re.compile(rf'\[{re.escape(var_name)}\s*\|\s*Seed\s*(\d+)\]\s*(Pre-train|DEC Fine-tune)\s*Epoch\s*(\d+)/(\d+)\s*\|\s*Loss:\s*([\d\.]+)(.*)')
            for line in var_text.split('\n'):
                m = pattern.search(line)
                if m:
                    seed = int(m.group(1))
                    phase = m.group(2)
                    ep = int(m.group(3))
                    tot_ep = int(m.group(4))
                    loss = float(m.group(5))
                    rest = m.group(6)
                    ari = float(re.search(r'ARI:\s*([\-\d\.]+)', rest).group(1)) if 'ARI:' in rest else None
                    nmi = float(re.search(r'NMI:\s*([\-\d\.]+)', rest).group(1)) if 'NMI:' in rest else None
                    sil = float(re.search(r'Sil:\s*([\-\d\.]+)', rest).group(1)) if 'Sil:' in rest else None
                    kl = float(re.search(r'KL:\s*([\-\d\.]+)', rest).group(1)) if 'KL:' in rest else None
                    
                    epoch_records.append({
                        'track': track_name,
                        'dataset': ds_name,
                        'variant': var_name,
                        'seed': seed,
                        'phase': phase,
                        'epoch': ep,
                        'total_epochs': tot_ep,
                        'loss': loss,
                        'kl': kl,
                        'ari': ari,
                        'nmi': nmi,
                        'sil': sil
                    })

    return seed_entries, epoch_records

def compute_aggregates(entries):
    variants = sorted(list(set(e['variant'] for e in entries)))
    datasets = sorted(list(set(e['dataset'] for e in entries)))
    
    var_stats = {}
    for v in variants:
        v_entries = [e for e in entries if e['variant'] == v]
        
        cats = ['dec_best_sil', 'dec_best_ari', 'dec_last_epoch', 'pretrain_best_sil', 'pretrain_best_ari']
        cat_stats = {}
        for c in cats:
            aris = [e['categories'][c]['ari'] for e in v_entries]
            sils = [e['categories'][c]['sil'] for e in v_entries]
            cat_stats[c] = {
                'ari_mean': float(np.mean(aris)),
                'ari_std': float(np.std(aris)),
                'sil_mean': float(np.mean(sils)),
                'sil_std': float(np.std(sils))
            }

        rts = [e['train_time_sec'] for e in v_entries]
        
        ds_map = {}
        for d in datasets:
            d_entries = [e for e in v_entries if e['dataset'] == d]
            ds_cat = {}
            for c in cats:
                d_aris = [e['categories'][c]['ari'] for e in d_entries]
                d_sils = [e['categories'][c]['sil'] for e in d_entries]
                ds_cat[c] = {
                    'ari_mean': float(np.mean(d_aris)) if d_aris else 0.0,
                    'ari_std': float(np.std(d_aris)) if d_aris else 0.0,
                    'sil_mean': float(np.mean(d_sils)) if d_sils else 0.0,
                    'sil_std': float(np.std(d_sils)) if d_sils else 0.0
                }
            ds_map[d] = {
                'categories': ds_cat,
                'seeds': [{
                    'seed': e['seed'],
                    'categories': e['categories'],
                    'runtime': e['train_time_sec']
                } for e in d_entries]
            }
            
        var_stats[v] = {
            'categories': cat_stats,
            'train_time_mean': float(np.mean(rts)),
            'train_time_std': float(np.std(rts)),
            'count': len(v_entries),
            'datasets': ds_map
        }
    return var_stats

def generate_html_page(page_type, title, subtitle, entries, epochs, output_file, root_rel=""):
    variants = sorted(list(set(e['variant'] for e in entries)))
    datasets = sorted(list(set(e['dataset'] for e in entries)))
    tracks = sorted(list(set(e['track'] for e in entries)))
    stats = compute_aggregates(entries)
    
    cat_defs = [
        {
            'id': 'dec_best_sil',
            'name': 'DEC Best Silhouette (Selected)',
            'short': 'DEC Best Sil (Selected)',
            'icon': '🎯',
            'color': '#6366f1',
            'desc': 'Unsupervised cluster quality selection during DEC fine-tuning (Primary: Selected ARI, Secondary: Silhouette).'
        },
        {
            'id': 'dec_best_ari',
            'name': 'DEC Best ARI Observed',
            'short': 'DEC Peak ARI',
            'icon': '🏆',
            'color': '#06b6d4',
            'desc': 'Maximum clustering ARI observed across all 200 DEC fine-tuning epochs.'
        },
        {
            'id': 'dec_last_epoch',
            'name': 'DEC Last Epoch (Terminal)',
            'short': 'DEC Terminal (Ep 200)',
            'icon': '🏁',
            'color': '#10b981',
            'desc': 'Final terminal state at epoch 200/200, evaluating model convergence stability.'
        },
        {
            'id': 'pretrain_best_sil',
            'name': 'Pre-train Best Silhouette',
            'short': 'Pre-train Best Sil',
            'icon': '🌱',
            'color': '#f59e0b',
            'desc': 'Best representation checkpoint selected by Silhouette during 250 pre-training epochs (used to initialize DEC centers).'
        },
        {
            'id': 'pretrain_best_ari',
            'name': 'Pre-train Best ARI',
            'short': 'Pre-train Peak ARI',
            'icon': '🌿',
            'color': '#ec4899',
            'desc': 'Peak clustering ARI achieved prior to DEC fine-tuning.'
        }
    ]

    best_dec_ari_entry = max(entries, key=lambda x: x['best_dec_ari'])
    best_dec_sil_entry = max(entries, key=lambda x: x['best_dec_sil'])
    best_pre_ari_entry = max(entries, key=lambda x: x['pretrain_best_ari'])
    best_pre_sil_entry = max(entries, key=lambda x: x['pretrain_best_sil'])
    
    best_variant_by_dec_ari = max(stats.items(), key=lambda x: x[1]['categories']['dec_best_ari']['ari_mean'])
    best_variant_by_dec_sil = max(stats.items(), key=lambda x: x[1]['categories']['dec_best_sil']['ari_mean'])
    
    entries_json = json.dumps(entries)
    stats_json = json.dumps(stats)
    epochs_json = json.dumps(epochs)
    datasets_json = json.dumps(datasets)
    variants_json = json.dumps(variants)
    cat_defs_json = json.dumps(cat_defs)
    
    nav_master = f"{root_rel}index.html"
    nav_encoder = f"{root_rel}encoder-ablation/index.html"
    nav_fusion = f"{root_rel}fusion-ablation/index.html"
    nav_loss = f"{root_rel}loss-ablation/index.html"
    nav_contrastive = f"{root_rel}contrastive-ablation/index.html"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | ARISE-V6 Ablation Analytics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-base: #080c16;
            --bg-surface: #0f172a;
            --bg-card: rgba(15, 23, 42, 0.78);
            --bg-card-hover: rgba(30, 41, 59, 0.88);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(99, 102, 241, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.28);
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --accent-violet: #8b5cf6;
            --accent-pink: #ec4899;
            --card-radius: 16px;
            --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-main);
            line-height: 1.5;
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.16) 0px, transparent 45%),
                radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.14) 0px, transparent 45%),
                radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.10) 0px, transparent 45%);
            background-attachment: fixed;
        }}

        /* Navbar */
        .navbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 36px;
            background: rgba(8, 12, 22, 0.88);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .nav-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: inherit;
        }}

        .brand-badge {{
            background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
            color: white;
            font-weight: 800;
            font-size: 0.82rem;
            padding: 6px 12px;
            border-radius: 8px;
            letter-spacing: 0.5px;
            box-shadow: 0 0 16px var(--primary-glow);
        }}

        .brand-title {{
            font-size: 1.15rem;
            font-weight: 700;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .nav-links {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(15, 23, 42, 0.65);
            padding: 4px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}

        .nav-item {{
            text-decoration: none;
            color: var(--text-muted);
            font-size: 0.88rem;
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}

        .nav-item:hover {{
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
        }}

        .nav-item.active {{
            color: #fff;
            background: var(--primary);
            box-shadow: 0 2px 10px var(--primary-glow);
        }}

        .container {{
            max-width: 1460px;
            margin: 0 auto;
            padding: 32px 24px 64px 24px;
        }}

        /* Hero Header */
        .hero {{
            margin-bottom: 28px;
        }}

        .hero-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .hero-title {{
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .hero-subtitle {{
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 950px;
        }}

        .tag-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(99, 102, 241, 0.14);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }}

        /* 5 CATEGORY SELECTOR BAR */
        .category-selector-wrapper {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--border-color);
            border-radius: var(--card-radius);
            padding: 16px 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }}

        .cat-selector-label {{
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-dim);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .cat-pills {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .cat-pill {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-color);
            padding: 10px 18px;
            border-radius: 12px;
            color: var(--text-muted);
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            user-select: none;
        }}

        .cat-pill:hover {{
            background: rgba(51, 65, 85, 0.7);
            color: #fff;
            transform: translateY(-1px);
        }}

        .cat-pill.active {{
            background: var(--primary);
            color: #fff;
            border-color: #818cf8;
            box-shadow: 0 4px 16px var(--primary-glow);
        }}

        .cat-pill.active.pill-cyan {{ background: var(--accent-cyan); border-color: #67e8f9; box-shadow: 0 4px 16px rgba(6,182,212,0.3); }}
        .cat-pill.active.pill-emerald {{ background: var(--accent-emerald); border-color: #6ee7b7; box-shadow: 0 4px 16px rgba(16,185,129,0.3); }}
        .cat-pill.active.pill-amber {{ background: var(--accent-amber); border-color: #fcd34d; color: #000; box-shadow: 0 4px 16px rgba(245,158,11,0.3); }}
        .cat-pill.active.pill-pink {{ background: var(--accent-pink); border-color: #f472b6; box-shadow: 0 4px 16px rgba(236,72,153,0.3); }}
        .cat-pill.active.pill-violet {{ background: var(--accent-violet); border-color: #c084fc; box-shadow: 0 4px 16px rgba(139,92,246,0.3); }}

        /* 5 CATEGORY STAT CARDS */
        .cat-cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 18px;
            margin-bottom: 32px;
        }}

        .cat-card {{
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-color);
            border-radius: var(--card-radius);
            padding: 20px;
            position: relative;
            overflow: hidden;
            transition: all 0.2s ease;
        }}

        .cat-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-hover);
        }}

        .cat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }}

        .cat-card.c-indigo::before {{ background: var(--primary); }}
        .cat-card.c-cyan::before {{ background: var(--accent-cyan); }}
        .cat-card.c-emerald::before {{ background: var(--accent-emerald); }}
        .cat-card.c-amber::before {{ background: var(--accent-amber); }}
        .cat-card.c-pink::before {{ background: var(--accent-pink); }}

        .cat-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }}

        .cat-card-title {{
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }}

        .cat-card-val {{
            font-size: 1.85rem;
            font-weight: 800;
            font-family: var(--font-mono);
            color: #fff;
            margin-bottom: 4px;
        }}

        .cat-card-sub {{
            font-size: 0.83rem;
            color: var(--text-dim);
            line-height: 1.4;
        }}

        .cat-highlight {{
            color: var(--accent-cyan);
            font-weight: 600;
        }}

        /* Tabs & Controls */
        .controls-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 22px;
            gap: 16px;
            flex-wrap: wrap;
        }}

        .tabs {{
            display: flex;
            gap: 8px;
            background: rgba(15, 23, 42, 0.75);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 0.88rem;
            font-weight: 600;
            font-family: var(--font-main);
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.04);
        }}

        .tab-btn.active {{
            color: white;
            background: var(--primary);
            box-shadow: 0 2px 10px var(--primary-glow);
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .select-input {{
            background: rgba(15, 23, 42, 0.85);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 0.88rem;
            font-family: var(--font-main);
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }}

        .select-input:hover, .select-input:focus {{
            border-color: var(--primary);
        }}

        /* Section Layouts */
        .tab-pane {{
            display: none;
        }}

        .tab-pane.active {{
            display: block;
            animation: fadeIn 0.25s ease forwards;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            gap: 24px;
            margin-bottom: 28px;
        }}

        .card {{
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-color);
            border-radius: var(--card-radius);
            padding: 24px;
        }}

        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }}

        .card-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .card-desc {{
            font-size: 0.84rem;
            color: var(--text-dim);
            margin-top: 2px;
        }}

        .chart-wrapper {{
            position: relative;
            height: 350px;
            width: 100%;
        }}

        /* Category Evolution & Gain Box */
        .evolution-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
            margin-bottom: 28px;
        }}

        .evolution-card {{
            background: rgba(30, 41, 59, 0.45);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
        }}

        .evolution-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .evolution-step {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 0.86rem;
        }}

        .evolution-step:last-child {{
            border-bottom: none;
        }}

        /* Data Tables */
        .table-controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .search-box {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 10px;
            color: white;
            font-size: 0.88rem;
            font-family: var(--font-main);
            width: 320px;
            outline: none;
        }}

        .search-box:focus {{
            border-color: var(--primary);
        }}

        .btn-action {{
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.84rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .btn-action:hover {{
            background: var(--primary);
            color: white;
        }}

        .table-container {{
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            text-align: left;
        }}

        th {{
            background: rgba(15, 23, 42, 0.95);
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}

        th:hover {{
            color: var(--text-main);
        }}

        td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-main);
            white-space: nowrap;
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.025);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: var(--font-mono);
        }}

        .badge-primary {{ background: rgba(99, 102, 241, 0.18); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }}
        .badge-cyan {{ background: rgba(6, 182, 212, 0.18); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.3); }}
        .badge-emerald {{ background: rgba(16, 185, 129, 0.18); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-amber {{ background: rgba(245, 158, 11, 0.18); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-pink {{ background: rgba(236, 72, 153, 0.18); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }}

        .metric-mono {{
            font-family: var(--font-mono);
            font-weight: 600;
        }}

        /* Dataset Matrix Heatmap */
        .matrix-table th, .matrix-table td {{
            text-align: center;
            padding: 12px;
        }}

        .matrix-table td:first-child {{
            text-align: left;
            font-weight: 600;
        }}

        .heat-cell {{
            border-radius: 8px;
            padding: 6px 12px;
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
            min-width: 80px;
        }}

        .footer {{
            text-align: center;
            margin-top: 48px;
            color: var(--text-dim);
            font-size: 0.82rem;
            border-top: 1px solid var(--border-color);
            padding-top: 24px;
        }}

        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            .navbar {{ padding: 14px 20px; flex-direction: column; gap: 12px; }}
            .hero-title {{ font-size: 1.75rem; }}
        }}
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header class="navbar">
        <a href="{nav_master}" class="nav-brand">
            <span class="brand-badge">ARISE-V6</span>
            <span class="brand-title">Ablation Intelligence</span>
        </a>
        <nav class="nav-links">
            <a href="{nav_master}" class="nav-item {'active' if page_type == 'master' else ''}">Master Suite</a>
            <a href="{nav_encoder}" class="nav-item {'active' if page_type == 'encoder' else ''}">Encoder Track</a>
            <a href="{nav_fusion}" class="nav-item {'active' if page_type == 'fusion' else ''}">Fusion Track</a>
            <a href="{nav_loss}" class="nav-item {'active' if page_type == 'loss' else ''}">Loss Track</a>
            <a href="{nav_contrastive}" class="nav-item {'active' if page_type == 'contrastive' else ''}">Contrastive Track</a>
        </nav>
    </header>

    <main class="container">
        <!-- Hero Section -->
        <section class="hero">
            <div class="hero-top">
                <span class="tag-pill">🎯 5-Category Benchmark Architecture</span>
                <span style="color: var(--text-dim); font-size: 0.85rem; font-family: var(--font-mono);">
                    {len(entries)} Total Runs • {len(variants)} Variants • {len(datasets)} Datasets
                </span>
            </div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </section>

        <!-- 5 CATEGORY SELECTOR / SWITCHER -->
        <section class="category-selector-wrapper">
            <div class="cat-selector-label">
                <span>Select Evaluation Category to Filter Dashboard:</span>
                <span id="currentCatDesc" style="color: var(--accent-cyan); font-weight: 500; text-transform: none;"></span>
            </div>
            <div class="cat-pills">
                <div class="cat-pill active pill-indigo" onclick="selectCategory('dec_best_sil', this)">
                    <span>🎯</span> <strong>DEC Best Silhouette (Selected)</strong>
                </div>
                <div class="cat-pill pill-cyan" onclick="selectCategory('dec_best_ari', this)">
                    <span>🏆</span> <strong>DEC Best ARI Observed</strong>
                </div>
                <div class="cat-pill pill-emerald" onclick="selectCategory('dec_last_epoch', this)">
                    <span>🏁</span> <strong>DEC Last Epoch (Terminal)</strong>
                </div>
                <div class="cat-pill pill-amber" onclick="selectCategory('pretrain_best_sil', this)">
                    <span>🌱</span> <strong>Pre-train Best Silhouette</strong>
                </div>
                <div class="cat-pill pill-pink" onclick="selectCategory('pretrain_best_ari', this)">
                    <span>🌿</span> <strong>Pre-train Best ARI</strong>
                </div>
                <div class="cat-pill pill-violet" onclick="selectCategory('all_compare', this)">
                    <span>📊</span> <strong>Compare All 5 Categories</strong>
                </div>
            </div>
        </section>

        <!-- 5 CATEGORY STAT CARDS -->
        <section class="cat-cards-grid">
            <div class="cat-card c-indigo">
                <div class="cat-card-header">
                    <span class="cat-card-title">🎯 DEC Best Sil (Selected)</span>
                    <span class="badge badge-primary">Selected ARI</span>
                </div>
                <div class="cat-card-val">{best_variant_by_dec_sil[1]['categories']['dec_best_sil']['ari_mean']:.4f}</div>
                <div class="cat-card-sub">Top Variant: <span class="cat-highlight">{best_variant_by_dec_sil[0]}</span><br>Sil Score: {best_variant_by_dec_sil[1]['categories']['dec_best_sil']['sil_mean']:.4f}</div>
            </div>

            <div class="cat-card c-cyan">
                <div class="cat-card-header">
                    <span class="cat-card-title">🏆 DEC Best ARI Observed</span>
                    <span class="badge badge-cyan">Peak DEC ARI</span>
                </div>
                <div class="cat-card-val">{best_variant_by_dec_ari[1]['categories']['dec_best_ari']['ari_mean']:.4f}</div>
                <div class="cat-card-sub">Top Variant: <span class="cat-highlight">{best_variant_by_dec_ari[0]}</span><br>Peak Run: {best_dec_ari_entry['best_dec_ari']:.4f} on {best_dec_ari_entry['dataset']}</div>
            </div>

            <div class="cat-card c-emerald">
                <div class="cat-card-header">
                    <span class="cat-card-title">🏁 DEC Last Epoch (Terminal)</span>
                    <span class="badge badge-emerald">Epoch 200/200</span>
                </div>
                <div class="cat-card-val">{np.mean([e['last_epoch_ari'] for e in entries]):.4f}</div>
                <div class="cat-card-sub">Terminal Mean ARI<br>Stability Index: {np.mean([e['last_epoch_ari'] / (e['best_dec_ari'] or 1) for e in entries]):.1%}</div>
            </div>

            <div class="cat-card c-amber">
                <div class="cat-card-header">
                    <span class="cat-card-title">🌱 Pre-train Best Sil</span>
                    <span class="badge badge-amber">Pre-train Initializer</span>
                </div>
                <div class="cat-card-val">{np.mean([e['pretrain_best_sil'] for e in entries]):.4f}</div>
                <div class="cat-card-sub">Mean Pre-train Sil<br>Corr ARI: {np.mean([e['pretrain_best_sil_corr_ari'] for e in entries]):.4f}</div>
            </div>

            <div class="cat-card c-pink">
                <div class="cat-card-header">
                    <span class="cat-card-title">🌿 Pre-train Best ARI</span>
                    <span class="badge badge-pink">Pre-train Peak</span>
                </div>
                <div class="cat-card-val">{np.mean([e['pretrain_best_ari'] for e in entries]):.4f}</div>
                <div class="cat-card-sub">Peak Run: <span class="cat-highlight">{best_pre_ari_entry['pretrain_best_ari']:.4f}</span><br>({best_pre_ari_entry['variant']})</div>
            </div>
        </section>

        <!-- Navigation Tabs & Interactive Controls -->
        <section class="controls-bar">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('overview')">📊 Category Performance</button>
                <button class="tab-btn" onclick="switchTab('progression')">⚡ Pre-train vs DEC Gain</button>
                <button class="tab-btn" onclick="switchTab('matrix')">🗺️ Dataset Matrix</button>
                <button class="tab-btn" onclick="switchTab('dynamics')">📈 Epoch Curves</button>
                <button class="tab-btn" onclick="switchTab('table')">📑 5-Category Full Table</button>
            </div>
            <div class="filter-group">
                <select id="metricValType" class="select-input" onchange="updateAllViews()">
                    <option value="ari">Display Metric: ARI Score</option>
                    <option value="sil">Display Metric: Silhouette Score</option>
                </select>
                <select id="datasetFilter" class="select-input" onchange="updateAllViews()">
                    <option value="ALL">All Datasets (Mean)</option>
                    {''.join(f'<option value="{d}">{d}</option>' for d in datasets)}
                </select>
            </div>
        </section>

        <!-- TAB 1: Category Performance Overview -->
        <div id="tab-overview" class="tab-pane active">
            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <h3 id="rankingChartTitle" class="card-title">🎯 DEC Best Silhouette (Selected) Ranking</h3>
                            <p id="rankingChartSub" class="card-desc">Mean performance across seeds and datasets</p>
                        </div>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="rankingChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div>
                            <h3 class="card-title">⚡ 5-Category Cross-Comparison</h3>
                            <p class="card-desc">Simultaneous view across all 5 evaluation stages per variant</p>
                        </div>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="allCatsCompareChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <h3 class="card-title">🧬 Multi-Category Radar Footprint</h3>
                            <p class="card-desc">Comparing Pre-training vs DEC Fine-tuning clustering profiles</p>
                        </div>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div>
                            <h3 class="card-title">⏱️ Training Time vs Selected Accuracy</h3>
                            <p class="card-desc">Evaluating computational overhead vs final clustering quality</p>
                        </div>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="paretoChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: Progression & Gain Analysis -->
        <div id="tab-progression" class="tab-pane">
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header">
                    <div>
                        <h3 id="gainChartTitle" class="card-title">🚀 DEC Fine-tuning Clustering Gain (ΔARI)</h3>
                        <p id="gainChartSub" class="card-desc">Quantifying ARI improvement from Pre-train Best Silhouette to DEC Best Silhouette (Selected)</p>
                    </div>
                </div>
                <div class="chart-wrapper" style="height: 380px;">
                    <canvas id="gainChart"></canvas>
                </div>
            </div>

            <div class="evolution-grid">
                <div class="evolution-card">
                    <div class="evolution-title">🎯 Unsupervised Selection Precision</div>
                    <p style="font-size:0.86rem; color:var(--text-muted); margin-bottom:12px;">
                        Comparing <strong>DEC Best Silhouette (Selected)</strong> ARI vs <strong>DEC Peak ARI Observed</strong> reveals whether internal silhouette optimization captures the ground-truth optimal cluster partition without label feedback.
                    </p>
                    <div class="evolution-step">
                        <span>Selected ARI Mean</span>
                        <strong id="cardSelectedAri" class="metric-mono" style="color:var(--accent-cyan);">-</strong>
                    </div>
                    <div class="evolution-step">
                        <span>Peak Observed ARI Mean</span>
                        <strong id="cardPeakAri" class="metric-mono" style="color:var(--primary);">-</strong>
                    </div>
                    <div class="evolution-step">
                        <span>Selection Efficiency</span>
                        <strong id="cardSelectionEff" class="metric-mono" style="color:var(--accent-emerald);">-</strong>
                    </div>
                </div>

                <div class="evolution-card">
                    <div class="evolution-title">🏁 Terminal Convergence Stability</div>
                    <p style="font-size:0.86rem; color:var(--text-muted); margin-bottom:12px;">
                        Comparing <strong>DEC Last Epoch (Terminal)</strong> ARI vs Peak ARI demonstrates if the model suffers from over-clustering collapse during late fine-tuning stages.
                    </p>
                    <div class="evolution-step">
                        <span>Terminal ARI Mean (Ep 200)</span>
                        <strong id="cardTerminalAri" class="metric-mono" style="color:var(--accent-emerald);">-</strong>
                    </div>
                    <div class="evolution-step">
                        <span>Peak ARI Mean</span>
                        <strong id="cardTerminalPeakAri" class="metric-mono" style="color:var(--accent-cyan);">-</strong>
                    </div>
                    <div class="evolution-step">
                        <span>Stability Retention</span>
                        <strong id="cardStabilityRet" class="metric-mono" style="color:var(--accent-amber);">-</strong>
                    </div>
                </div>

                <div class="evolution-card">
                    <div class="evolution-title">🌱 Pre-train Representation Quality</div>
                    <p style="font-size:0.86rem; color:var(--text-muted); margin-bottom:12px;">
                        Pre-training representation baseline before DEC soft assignment optimization.
                    </p>
                    <div class="evolution-step">
                        <span>Pre-train Best Sil Mean</span>
                        <strong id="cardPreSilMean" class="metric-mono" style="color:var(--accent-amber);">-</strong>
                    </div>
                    <div class="evolution-step">
                        <span>Pre-train Best ARI Mean</span>
                        <strong id="cardPreAriMean" class="metric-mono" style="color:var(--accent-pink);">-</strong>
                    </div>
                    <div class="evolution-step">
                        <span>DEC Fine-tune Net Boost</span>
                        <strong id="cardNetBoost" class="metric-mono" style="color:var(--accent-cyan);">-</strong>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 3: Dataset Matrix Heatmap -->
        <div id="tab-matrix" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <div>
                        <h3 id="matrixTitle" class="card-title">Dataset Performance Matrix: 🎯 DEC Best Silhouette (Selected)</h3>
                        <p class="card-desc">Dynamic heatmap showing metric distribution (Mean ± Std) across tissue sections</p>
                    </div>
                </div>
                <div class="table-container">
                    <table class="matrix-table">
                        <thead>
                            <tr>
                                <th>Variant</th>
                                {''.join(f'<th>{d}</th>' for d in datasets)}
                                <th>Overall Mean</th>
                            </tr>
                        </thead>
                        <tbody id="matrixTableBody">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 4: Epoch Dynamics -->
        <div id="tab-dynamics" class="tab-pane">
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header" style="flex-wrap: wrap; gap: 12px;">
                    <div>
                        <h3 class="card-title">Training Loss & Clustering Metric Trajectory</h3>
                        <p class="card-desc">Pre-training reconstruction loss decay & DEC fine-tuning convergence curves</p>
                    </div>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <select id="epochVariantSelect" class="select-input" onchange="renderEpochChart()">
                            {''.join(f'<option value="{v}">{v}</option>' for v in variants)}
                        </select>
                        <select id="epochDatasetSelect" class="select-input" onchange="renderEpochChart()">
                            {''.join(f'<option value="{d}">{d}</option>' for d in datasets)}
                        </select>
                        <select id="epochSeedSelect" class="select-input" onchange="renderEpochChart()">
                            <option value="42">Seed 42</option>
                            <option value="1234">Seed 1234</option>
                            <option value="2024">Seed 2024</option>
                        </select>
                    </div>
                </div>
                <div class="chart-wrapper" style="height: 420px;">
                    <canvas id="epochChart"></canvas>
                </div>
            </div>
        </div>

        <!-- TAB 5: Searchable 5-Category Data Table -->
        <div id="tab-table" class="tab-pane">
            <div class="card">
                <div class="table-controls">
                    <input type="text" id="tableSearch" class="search-box" placeholder="🔍 Filter variant, dataset, seed..." oninput="filterTable()">
                    <div style="display: flex; gap: 8px;">
                        <button class="btn-action" onclick="exportCSV()">📥 Export CSV</button>
                        <button class="btn-action" onclick="exportJSON()">📦 Export JSON</button>
                    </div>
                </div>
                <div class="table-container">
                    <table id="rawDataTable">
                        <thead>
                            <tr>
                                <th onclick="sortTable(0)">Track ↕</th>
                                <th onclick="sortTable(1)">Variant ↕</th>
                                <th onclick="sortTable(2)">Dataset ↕</th>
                                <th onclick="sortTable(3)">Seed ↕</th>
                                <th onclick="sortTable(4)" title="DEC Best Silhouette Selected ARI">🎯 DEC Best Sil ARI (Epoch) ↕</th>
                                <th onclick="sortTable(5)" title="DEC Peak ARI Observed">🏆 DEC Best ARI (Epoch) ↕</th>
                                <th onclick="sortTable(6)" title="DEC Terminal State at Epoch 200">🏁 DEC Last Ep ARI ↕</th>
                                <th onclick="sortTable(7)" title="Pre-train Silhouette Selected ARI">🌱 Pre-train Sil (Corr ARI) ↕</th>
                                <th onclick="sortTable(8)" title="Pre-train Peak ARI Observed">🌿 Pre-train Peak ARI ↕</th>
                                <th onclick="sortTable(9)">Runtime (s) ↕</th>
                            </tr>
                        </thead>
                        <tbody id="dataTableBody">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="footer">
            ARISE-V6 Spatial Omics Multi-Modal Framework • 5-Category Ablation Analytics Suite • Generated Automatically
        </footer>
    </main>

    <script>
        const RAW_ENTRIES = {entries_json};
        const STATS_DATA = {stats_json};
        const EPOCHS_DATA = {epochs_json};
        const DATASETS = {datasets_json};
        const VARIANTS = {variants_json};
        const CAT_DEFS = {cat_defs_json};

        let currentCategory = 'dec_best_sil';
        let rankingChartInst = null;
        let allCatsCompareChartInst = null;
        let radarChartInst = null;
        let paretoChartInst = null;
        let gainChartInst = null;
        let epochChartInst = null;

        const COLOR_PALETTE = [
            '#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#14b8a6', '#f43f5e', '#3b82f6'
        ];

        function selectCategory(catId, element) {{
            currentCategory = catId;
            document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
            if (element) element.classList.add('active');

            const def = CAT_DEFS.find(c => c.id === catId);
            const descEl = document.getElementById('currentCatDesc');
            if (descEl) {{
                descEl.innerText = def ? def.desc : (catId === 'all_compare' ? 'Side-by-side comparison across all 5 evaluation stages.' : '');
            }}

            const matrixTitle = document.getElementById('matrixTitle');
            if (matrixTitle) {{
                matrixTitle.innerText = def ? `Dataset Performance Matrix: ${{def.icon}} ${{def.name}}` : 'Dataset Performance Matrix (All 5 Categories)';
            }}

            const rankTitle = document.getElementById('rankingChartTitle');
            if (rankTitle) {{
                rankTitle.innerText = def ? `${{def.icon}} ${{def.name}} Ranking` : 'Variant Ranking (Selected Silhouette)';
            }}

            updateAllViews();
        }}

        function updateAllViews() {{
            updateCharts();
            renderGainChart();
            renderMatrix();
            updateEvolutionCards();
        }}

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            
            event.target.classList.add('active');
            const targetPane = document.getElementById('tab-' + tabId);
            if (targetPane) targetPane.classList.add('active');

            if (tabId === 'overview') {{
                setTimeout(updateCharts, 50);
            }} else if (tabId === 'progression') {{
                setTimeout(() => {{
                    renderGainChart();
                    updateEvolutionCards();
                }}, 50);
            }} else if (tabId === 'matrix') {{
                renderMatrix();
            }} else if (tabId === 'dynamics') {{
                setTimeout(renderEpochChart, 50);
            }} else if (tabId === 'table') {{
                renderTable(RAW_ENTRIES);
            }}
        }}

        function updateCharts() {{
            const valType = document.getElementById('metricValType').value; // 'ari' or 'sil'
            const dsFilter = document.getElementById('datasetFilter').value;

            // 1. Ranking Chart based on selected category
            const rankingLabels = [];
            const rankingValues = [];

            const activeCat = (currentCategory === 'all_compare') ? 'dec_best_sil' : currentCategory;

            VARIANTS.forEach(v => {{
                rankingLabels.push(v);
                let relevant = RAW_ENTRIES.filter(e => e.variant === v);
                if (dsFilter !== 'ALL') {{
                    relevant = relevant.filter(e => e.dataset === dsFilter);
                }}
                const vals = relevant.map(e => e.categories[activeCat][valType]);
                const mean = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
                rankingValues.push(mean);
            }});

            const ctxRanking = document.getElementById('rankingChart').getContext('2d');
            if (rankingChartInst) rankingChartInst.destroy();
            rankingChartInst = new Chart(ctxRanking, {{
                type: 'bar',
                data: {{
                    labels: rankingLabels,
                    datasets: [{{
                        label: activeCat.toUpperCase() + ' (' + valType.toUpperCase() + ')',
                        data: rankingValues,
                        backgroundColor: rankingLabels.map((_, i) => COLOR_PALETTE[i % COLOR_PALETTE.length] + 'cc'),
                        borderColor: rankingLabels.map((_, i) => COLOR_PALETTE[i % COLOR_PALETTE.length]),
                        borderWidth: 1.5,
                        borderRadius: 6
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#94a3b8' }} }}
                    }}
                }}
            }});

            // 2. All 5 Categories Side-by-Side Grouped Chart
            const ctxAll = document.getElementById('allCatsCompareChart').getContext('2d');
            if (allCatsCompareChartInst) allCatsCompareChartInst.destroy();

            const allCatDatasets = CAT_DEFS.map((c, i) => {{
                const vals = VARIANTS.map(v => {{
                    let relevant = RAW_ENTRIES.filter(e => e.variant === v);
                    if (dsFilter !== 'ALL') relevant = relevant.filter(e => e.dataset === dsFilter);
                    const scores = relevant.map(e => e.categories[c.id][valType]);
                    return scores.length ? (scores.reduce((a,b)=>a+b,0)/scores.length) : 0;
                }});
                return {{
                    label: c.short,
                    data: vals,
                    backgroundColor: c.color + 'cc',
                    borderColor: c.color,
                    borderWidth: 1,
                    borderRadius: 4
                }};
            }});

            allCatsCompareChartInst = new Chart(ctxAll, {{
                type: 'bar',
                data: {{
                    labels: VARIANTS,
                    datasets: allCatDatasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', boxWidth: 10, font: {{ size: 10 }} }} }}
                    }},
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#94a3b8' }} }},
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#94a3b8' }} }}
                    }}
                }}
            }});

            // 3. Multi-Category Radar
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            if (radarChartInst) radarChartInst.destroy();

            const radarCats = ['dec_best_sil', 'dec_best_ari', 'dec_last_epoch', 'pretrain_best_sil', 'pretrain_best_ari'];
            const radarLabels = ['🎯 DEC Best Sil', '🏆 DEC Peak ARI', '🏁 DEC Last Ep', '🌱 Pre-train Sil', '🌿 Pre-train ARI'];

            const radarDatasets = VARIANTS.map((v, i) => {{
                const data = radarCats.map(c => {{
                    let relevant = RAW_ENTRIES.filter(e => e.variant === v);
                    if (dsFilter !== 'ALL') relevant = relevant.filter(e => e.dataset === dsFilter);
                    const scores = relevant.map(e => e.categories[c][valType]);
                    return scores.length ? (scores.reduce((a,b)=>a+b,0)/scores.length) : 0;
                }});
                return {{
                    label: v,
                    data: data,
                    borderColor: COLOR_PALETTE[i % COLOR_PALETTE.length],
                    backgroundColor: COLOR_PALETTE[i % COLOR_PALETTE.length] + '20',
                    pointBackgroundColor: COLOR_PALETTE[i % COLOR_PALETTE.length],
                    borderWidth: 2
                }};
            }});

            radarChartInst = new Chart(ctxRadar, {{
                type: 'radar',
                data: {{ labels: radarLabels, datasets: radarDatasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        r: {{
                            grid: {{ color: 'rgba(255,255,255,0.08)' }},
                            angleLines: {{ color: 'rgba(255,255,255,0.08)' }},
                            pointLabels: {{ color: '#94a3b8', font: {{ size: 10, weight: 600 }} }},
                            ticks: {{ color: '#64748b', backdropColor: 'transparent' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', boxWidth: 10, font: {{ size: 10 }} }} }}
                    }}
                }}
            }});

            // 4. Pareto Scatter Chart
            const ctxPareto = document.getElementById('paretoChart').getContext('2d');
            if (paretoChartInst) paretoChartInst.destroy();

            const paretoPoints = VARIANTS.map((v, i) => {{
                let relevant = RAW_ENTRIES.filter(e => e.variant === v);
                if (dsFilter !== 'ALL') relevant = relevant.filter(e => e.dataset === dsFilter);
                const aris = relevant.map(e => e.categories[activeCat].ari);
                const rts = relevant.map(e => e.train_time_sec);
                const meanAri = aris.length ? (aris.reduce((a,b)=>a+b,0)/aris.length) : 0;
                const meanRt = rts.length ? (rts.reduce((a,b)=>a+b,0)/rts.length) : 0;

                return {{
                    label: v,
                    data: [{{ x: meanRt, y: meanAri }}],
                    backgroundColor: COLOR_PALETTE[i % COLOR_PALETTE.length],
                    borderColor: '#fff',
                    borderWidth: 1.5,
                    pointRadius: 8,
                    pointHoverRadius: 11
                }};
            }});

            paretoChartInst = new Chart(ctxPareto, {{
                type: 'scatter',
                data: {{ datasets: paretoPoints }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', boxWidth: 10, font: {{ size: 10 }} }} }},
                        tooltip: {{
                            callbacks: {{
                                label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x.toFixed(1) + 's, ARI: ' + ctx.parsed.y.toFixed(4)
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Mean Runtime (seconds)', color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                        y: {{ title: {{ display: true, text: 'Mean Selected ARI', color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }}
                    }}
                }}
            }});
        }}

        // Dynamic Progression & Gain Chart based on active category & dataset filter
        function renderGainChart() {{
            const ctxGain = document.getElementById('gainChart').getContext('2d');
            if (gainChartInst) gainChartInst.destroy();

            const dsFilter = document.getElementById('datasetFilter').value;
            const valType = document.getElementById('metricValType').value;
            const titleEl = document.getElementById('gainChartTitle');
            const subEl = document.getElementById('gainChartSub');

            let baselineLabel = 'Pre-train Best Sil';
            let targetLabel = 'DEC Best Silhouette (Selected)';
            let baselineKey = 'pretrain_best_sil';
            let targetKey = 'dec_best_sil';

            if (currentCategory === 'dec_best_ari') {{
                baselineLabel = 'Pre-train Peak ARI (Baseline)';
                targetLabel = 'DEC Best ARI Observed (Peak)';
                baselineKey = 'pretrain_best_ari';
                targetKey = 'dec_best_ari';
                if (titleEl) titleEl.innerText = '🏆 DEC Peak ARI Upper-Bound Gain (ΔARI)';
                if (subEl) subEl.innerText = 'Quantifying peak optimization gain from Pre-train Peak ARI to DEC Best ARI Observed';
            }} else if (currentCategory === 'dec_last_epoch') {{
                baselineLabel = 'Pre-train Best Sil (Baseline)';
                targetLabel = 'DEC Last Epoch (Terminal Ep 200)';
                baselineKey = 'pretrain_best_sil';
                targetKey = 'dec_last_epoch';
                if (titleEl) titleEl.innerText = '🏁 DEC Terminal Convergence Stability Delta (ΔARI)';
                if (subEl) subEl.innerText = 'Evaluating final terminal epoch performance relative to initial pre-training representation';
            }} else if (currentCategory === 'pretrain_best_sil') {{
                baselineLabel = 'Pre-train Selected Sil Score';
                targetLabel = 'Pre-train Corresponding ARI';
                baselineKey = 'pretrain_best_sil';
                targetKey = 'pretrain_best_sil';
                if (titleEl) titleEl.innerText = '🌱 Pre-train Silhouette Representation Baseline';
                if (subEl) subEl.innerText = 'Evaluating pre-training silhouette separation and corresponding ground-truth cluster overlap';
            }} else if (currentCategory === 'pretrain_best_ari') {{
                baselineLabel = 'Pre-train Best Sil ARI';
                targetLabel = 'Pre-train Peak ARI Observed';
                baselineKey = 'pretrain_best_sil';
                targetKey = 'pretrain_best_ari';
                if (titleEl) titleEl.innerText = '🌿 Pre-train Peak vs Selected ARI Range';
                if (subEl) subEl.innerText = 'Comparing initial representation spread before DEC soft-assignment optimization';
            }} else if (currentCategory === 'all_compare') {{
                if (titleEl) titleEl.innerText = '📊 All 5-Category Complete Progression';
                if (subEl) subEl.innerText = 'Comprehensive multi-stage progression from Pre-training through DEC Fine-tuning to Terminal State';
            }} else {{
                // Default dec_best_sil
                if (titleEl) titleEl.innerText = '🎯 DEC Fine-tuning Clustering Gain (ΔARI)';
                if (subEl) subEl.innerText = 'Quantifying unsupervised ARI improvement from Pre-train Best Silhouette to DEC Best Silhouette (Selected)';
            }}

            if (currentCategory === 'all_compare') {{
                // Grouped bar with all 5 stages
                const datasets = CAT_DEFS.map((c, i) => {{
                    const vals = VARIANTS.map(v => {{
                        let rel = RAW_ENTRIES.filter(e => e.variant === v);
                        if (dsFilter !== 'ALL') rel = rel.filter(e => e.dataset === dsFilter);
                        const sc = rel.map(e => e.categories[c.id][valType]);
                        return sc.length ? (sc.reduce((a,b)=>a+b,0)/sc.length) : 0;
                    }});
                    return {{
                        label: c.name,
                        data: vals,
                        backgroundColor: c.color + 'cc',
                        borderColor: c.color,
                        borderWidth: 1,
                        borderRadius: 4
                    }};
                }});

                gainChartInst = new Chart(ctxGain, {{
                    type: 'bar',
                    data: {{ labels: VARIANTS, datasets: datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
                        scales: {{
                            x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#94a3b8' }} }},
                            y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#94a3b8' }} }}
                        }}
                    }}
                }});
                return;
            }}

            const baselineVals = VARIANTS.map(v => {{
                let rel = RAW_ENTRIES.filter(e => e.variant === v);
                if (dsFilter !== 'ALL') rel = rel.filter(e => e.dataset === dsFilter);
                const vals = rel.map(e => (currentCategory === 'pretrain_best_sil' ? e.categories[baselineKey].sil : e.categories[baselineKey][valType]));
                return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length) : 0;
            }});

            const targetVals = VARIANTS.map(v => {{
                let rel = RAW_ENTRIES.filter(e => e.variant === v);
                if (dsFilter !== 'ALL') rel = rel.filter(e => e.dataset === dsFilter);
                const vals = rel.map(e => e.categories[targetKey][valType]);
                return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length) : 0;
            }});

            const deltas = targetVals.map((t, i) => t - baselineVals[i]);

            gainChartInst = new Chart(ctxGain, {{
                type: 'bar',
                data: {{
                    labels: VARIANTS,
                    datasets: [
                        {{
                            label: baselineLabel,
                            data: baselineVals,
                            backgroundColor: 'rgba(245, 158, 11, 0.75)',
                            borderColor: '#f59e0b',
                            borderWidth: 1,
                            borderRadius: 4
                        }},
                        {{
                            label: targetLabel,
                            data: targetVals,
                            backgroundColor: 'rgba(6, 182, 212, 0.85)',
                            borderColor: '#06b6d4',
                            borderWidth: 1,
                            borderRadius: 4
                        }},
                        {{
                            label: 'Net Delta (Δ' + valType.toUpperCase() + ')',
                            data: deltas,
                            backgroundColor: deltas.map(d => d >= 0 ? 'rgba(16, 185, 129, 0.85)' : 'rgba(244, 63, 94, 0.85)'),
                            borderColor: deltas.map(d => d >= 0 ? '#10b981' : '#f43f5e'),
                            borderWidth: 1,
                            borderRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}
                    }},
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#94a3b8' }} }},
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#94a3b8' }} }}
                    }}
                }}
            }});
        }}

        // Dynamic Evolution Stats Cards
        function updateEvolutionCards() {{
            const dsFilter = document.getElementById('datasetFilter').value;
            let rel = RAW_ENTRIES;
            if (dsFilter !== 'ALL') rel = rel.filter(e => e.dataset === dsFilter);

            const selAris = rel.map(e => e.categories.dec_best_sil.ari);
            const peakAris = rel.map(e => e.categories.dec_best_ari.ari);
            const termAris = rel.map(e => e.categories.dec_last_epoch.ari);
            const preSils = rel.map(e => e.categories.pretrain_best_sil.sil);
            const preSilAris = rel.map(e => e.categories.pretrain_best_sil.ari);
            const prePeakAris = rel.map(e => e.categories.pretrain_best_ari.ari);

            const meanSel = selAris.length ? (selAris.reduce((a,b)=>a+b,0)/selAris.length) : 0;
            const meanPeak = peakAris.length ? (peakAris.reduce((a,b)=>a+b,0)/peakAris.length) : 0;
            const meanTerm = termAris.length ? (termAris.reduce((a,b)=>a+b,0)/termAris.length) : 0;
            const meanPreSil = preSils.length ? (preSils.reduce((a,b)=>a+b,0)/preSils.length) : 0;
            const meanPreSilAri = preSilAris.length ? (preSilAris.reduce((a,b)=>a+b,0)/preSilAris.length) : 0;
            const meanPrePeakAri = prePeakAris.length ? (prePeakAris.reduce((a,b)=>a+b,0)/prePeakAris.length) : 0;

            const selEff = meanPeak > 0 ? (meanSel / meanPeak * 100).toFixed(1) + '%' : '100%';
            const stabRet = meanPeak > 0 ? (meanTerm / meanPeak * 100).toFixed(1) + '%' : '100%';
            const netBoost = (meanSel - meanPreSilAri);

            const elSel = document.getElementById('cardSelectedAri');
            if (elSel) elSel.innerText = meanSel.toFixed(4);

            const elPeak = document.getElementById('cardPeakAri');
            if (elPeak) elPeak.innerText = meanPeak.toFixed(4);

            const elEff = document.getElementById('cardSelectionEff');
            if (elEff) elEff.innerText = selEff;

            const elTerm = document.getElementById('cardTerminalAri');
            if (elTerm) elTerm.innerText = meanTerm.toFixed(4);

            const elTermPeak = document.getElementById('cardTerminalPeakAri');
            if (elTermPeak) elTermPeak.innerText = meanPeak.toFixed(4);

            const elStab = document.getElementById('cardStabilityRet');
            if (elStab) elStab.innerText = stabRet;

            const elPreSil = document.getElementById('cardPreSilMean');
            if (elPreSil) elPreSil.innerText = meanPreSil.toFixed(4);

            const elPreAri = document.getElementById('cardPreAriMean');
            if (elPreAri) elPreAri.innerText = meanPrePeakAri.toFixed(4);

            const elBoost = document.getElementById('cardNetBoost');
            if (elBoost) elBoost.innerText = (netBoost >= 0 ? '+' : '') + netBoost.toFixed(4) + ' ARI';
        }}

        // Matrix Rendering
        function renderMatrix() {{
            const tbody = document.getElementById('matrixTableBody');
            tbody.innerHTML = '';
            
            const activeCat = (currentCategory === 'all_compare') ? 'dec_best_sil' : currentCategory;
            const valType = document.getElementById('metricValType').value;

            let allVals = [];
            VARIANTS.forEach(v => {{
                DATASETS.forEach(d => {{
                    const st = STATS_DATA[v]?.datasets[d]?.categories[activeCat];
                    if (st) allVals.push(st[valType + '_mean']);
                }});
            }});
            const minV = Math.min(...allVals);
            const maxV = Math.max(...allVals);

            function getHeatColor(val) {{
                const ratio = Math.max(0, Math.min(1, (val - minV) / (maxV - minV || 1)));
                const r = Math.round(15 + ratio * 84);
                const g = Math.round(23 + ratio * 159);
                const b = Math.round(42 + ratio * 200);
                return `rgba(${{r}}, ${{g}}, ${{b}}, ${{0.35 + ratio * 0.55}})`;
            }}

            VARIANTS.forEach(v => {{
                const tr = document.createElement('tr');
                let rowHtml = `<td><span class="badge badge-primary">${{v}}</span></td>`;
                DATASETS.forEach(d => {{
                    const st = STATS_DATA[v]?.datasets[d]?.categories[activeCat];
                    const val = st ? st[valType + '_mean'] : 0;
                    const std = st ? st[valType + '_std'] : 0;
                    rowHtml += `<td><span class="heat-cell" style="background:${{getHeatColor(val)}}">${{val.toFixed(4)}} <small style="color:var(--text-dim)">±${{std.toFixed(3)}}</small></span></td>`;
                }});
                const overallMean = STATS_DATA[v]?.categories[activeCat]?.[valType + '_mean'] || 0;
                rowHtml += `<td><strong>${{overallMean.toFixed(4)}}</strong></td>`;
                tr.innerHTML = rowHtml;
                tbody.appendChild(tr);
            }});
        }}

        // Epoch Trajectory Chart
        function renderEpochChart() {{
            const variant = document.getElementById('epochVariantSelect').value;
            const dataset = document.getElementById('epochDatasetSelect').value;
            const seed = parseInt(document.getElementById('epochSeedSelect').value);

            const filteredEpochs = EPOCHS_DATA.filter(e => 
                e.variant === variant && e.dataset === dataset && e.seed === seed
            );

            const pretrain = filteredEpochs.filter(e => e.phase === 'Pre-train').sort((a,b) => a.epoch - b.epoch);
            const dec = filteredEpochs.filter(e => e.phase === 'DEC Fine-tune').sort((a,b) => a.epoch - b.epoch);

            const labels = [];
            const lossData = [];
            const ariData = [];
            const silData = [];

            pretrain.forEach(p => {{
                labels.push(`Pre-${{p.epoch}}`);
                lossData.push(p.loss);
                ariData.push(p.ari);
                silData.push(p.sil);
            }});

            dec.forEach(d => {{
                labels.push(`DEC-${{d.epoch}}`);
                lossData.push(d.loss);
                ariData.push(d.ari);
                silData.push(d.sil);
            }});

            const ctx = document.getElementById('epochChart').getContext('2d');
            if (epochChartInst) epochChartInst.destroy();

            epochChartInst = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Loss',
                            data: lossData,
                            borderColor: '#f59e0b',
                            backgroundColor: '#f59e0b22',
                            yAxisID: 'yLoss',
                            tension: 0.25,
                            pointRadius: 4
                        }},
                        {{
                            label: 'ARI Metric',
                            data: ariData,
                            borderColor: '#06b6d4',
                            backgroundColor: '#06b6d422',
                            yAxisID: 'yMetrics',
                            tension: 0.25,
                            pointRadius: 4
                        }},
                        {{
                            label: 'Silhouette Score',
                            data: silData,
                            borderColor: '#10b981',
                            backgroundColor: '#10b98122',
                            yAxisID: 'yMetrics',
                            tension: 0.25,
                            pointRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#94a3b8' }} }},
                        yLoss: {{
                            type: 'linear',
                            position: 'left',
                            title: {{ display: true, text: 'Loss', color: '#f59e0b' }},
                            grid: {{ color: 'rgba(255,255,255,0.05)' }},
                            ticks: {{ color: '#f59e0b' }}
                        }},
                        yMetrics: {{
                            type: 'linear',
                            position: 'right',
                            title: {{ display: true, text: 'ARI / Silhouette', color: '#06b6d4' }},
                            grid: {{ drawOnChartArea: false }},
                            ticks: {{ color: '#06b6d4' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}
                    }}
                }}
            }});
        }}

        // Data Table Rendering
        let currentSortCol = -1;
        let isAscending = true;

        function renderTable(data) {{
            const tbody = document.getElementById('dataTableBody');
            tbody.innerHTML = '';
            
            data.forEach(e => {{
                const c = e.categories;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><span class="badge badge-${{e.track === 'Encoder' ? 'emerald' : e.track === 'Loss' ? 'amber' : e.track === 'Contrastive' ? 'pink' : 'primary'}}">${{e.track}}</span></td>
                    <td><strong>${{e.variant}}</strong></td>
                    <td>${{e.dataset}}</td>
                    <td><span class="badge badge-cyan">${{e.seed}}</span></td>
                    <td class="metric-mono" style="color:#818cf8;">${{c.dec_best_sil.ari.toFixed(4)}} <small style="color:var(--text-dim)">(${{c.dec_best_sil.epoch}})</small></td>
                    <td class="metric-mono" style="color:var(--accent-cyan); font-weight:700;">${{c.dec_best_ari.ari.toFixed(4)}} <small style="color:var(--text-dim)">(${{c.dec_best_ari.epoch}})</small></td>
                    <td class="metric-mono" style="color:var(--accent-emerald);">${{c.dec_last_epoch.ari.toFixed(4)}}</td>
                    <td class="metric-mono" style="color:var(--accent-amber);">${{c.pretrain_best_sil.sil.toFixed(4)}} <small style="color:var(--text-dim)">(ARI: ${{c.pretrain_best_sil.ari.toFixed(4)}})</small></td>
                    <td class="metric-mono" style="color:var(--accent-pink);">${{c.pretrain_best_ari.ari.toFixed(4)}} <small style="color:var(--text-dim)">(${{c.pretrain_best_ari.epoch}})</small></td>
                    <td class="metric-mono" style="color:var(--text-muted);">${{e.train_time_sec.toFixed(1)}}s</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function filterTable() {{
            const q = document.getElementById('tableSearch').value.toLowerCase();
            const filtered = RAW_ENTRIES.filter(e => 
                e.variant.toLowerCase().includes(q) ||
                e.dataset.toLowerCase().includes(q) ||
                e.track.toLowerCase().includes(q) ||
                String(e.seed).includes(q)
            );
            renderTable(filtered);
        }}

        function sortTable(colIndex) {{
            if (currentSortCol === colIndex) {{
                isAscending = !isAscending;
            }} else {{
                currentSortCol = colIndex;
                isAscending = true;
            }}

            RAW_ENTRIES.sort((a, b) => {{
                let va, vb;
                if (colIndex === 0) {{ va = a.track; vb = b.track; }}
                else if (colIndex === 1) {{ va = a.variant; vb = b.variant; }}
                else if (colIndex === 2) {{ va = a.dataset; vb = b.dataset; }}
                else if (colIndex === 3) {{ va = a.seed; vb = b.seed; }}
                else if (colIndex === 4) {{ va = a.categories.dec_best_sil.ari; vb = b.categories.dec_best_sil.ari; }}
                else if (colIndex === 5) {{ va = a.categories.dec_best_ari.ari; vb = b.categories.dec_best_ari.ari; }}
                else if (colIndex === 6) {{ va = a.categories.dec_last_epoch.ari; vb = b.categories.dec_last_epoch.ari; }}
                else if (colIndex === 7) {{ va = a.categories.pretrain_best_sil.sil; vb = b.categories.pretrain_best_sil.sil; }}
                else if (colIndex === 8) {{ va = a.categories.pretrain_best_ari.ari; vb = b.categories.pretrain_best_ari.ari; }}
                else if (colIndex === 9) {{ va = a.train_time_sec; vb = b.train_time_sec; }}

                if (typeof va === 'string') {{
                    return isAscending ? va.localeCompare(vb) : vb.localeCompare(va);
                }}
                return isAscending ? va - vb : vb - va;
            }});

            filterTable();
        }}

        function exportCSV() {{
            const headers = [
                'Track', 'Variant', 'Dataset', 'Seed', 
                'DEC_Best_Sil_Epoch', 'DEC_Best_Sil_Silhouette', 'DEC_Best_Sil_Selected_ARI',
                'DEC_Best_ARI_Epoch', 'DEC_Best_ARI_Observed', 'DEC_Best_ARI_Corr_Sil',
                'DEC_Last_Epoch_ARI', 'DEC_Last_Epoch_Sil',
                'Pretrain_Best_Sil_Epoch', 'Pretrain_Best_Sil', 'Pretrain_Best_Sil_Corr_ARI',
                'Pretrain_Best_ARI_Epoch', 'Pretrain_Best_ARI', 'Pretrain_Best_ARI_Corr_Sil',
                'Runtime_Sec'
            ];
            const rows = RAW_ENTRIES.map(e => {{
                const c = e.categories;
                return [
                    e.track, e.variant, e.dataset, e.seed,
                    c.dec_best_sil.epoch, c.dec_best_sil.sil, c.dec_best_sil.ari,
                    c.dec_best_ari.epoch, c.dec_best_ari.ari, c.dec_best_ari.sil,
                    c.dec_last_epoch.ari, c.dec_last_epoch.sil,
                    c.pretrain_best_sil.epoch, c.pretrain_best_sil.sil, c.pretrain_best_sil.ari,
                    c.pretrain_best_ari.epoch, c.pretrain_best_ari.ari, c.pretrain_best_ari.sil,
                    e.train_time_sec
                ];
            }});
            let csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\\n');
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement('a');
            link.setAttribute('href', encodedUri);
            link.setAttribute('download', 'arise_v6_5categories_benchmark.csv');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        function exportJSON() {{
            const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(RAW_ENTRIES, null, 2));
            const link = document.createElement('a');
            link.setAttribute('href', dataStr);
            link.setAttribute('download', 'arise_v6_5categories_benchmark.json');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            selectCategory('dec_best_sil', document.querySelector('.cat-pill'));
            renderTable(RAW_ENTRIES);
            renderMatrix();
            renderGainChart();
            updateEvolutionCards();
        }});
    </script>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated 5-category dashboard: {output_file}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    enc_log = os.path.join(base_dir, 'encoder-ablation', 'log.txt')
    fus_log = os.path.join(base_dir, 'fusion-ablation', 'log.txt')
    loss_log = os.path.join(base_dir, 'loss-ablation', 'log.txt')
    cont_log = os.path.join(base_dir, 'contrastive-ablation', 'log.txt')

    print("Parsing logs for 5-category evaluation...")
    enc_entries, enc_epochs = parse_ablation_log(enc_log, 'Encoder')
    fus_entries, fus_epochs = parse_ablation_log(fus_log, 'Fusion')
    loss_entries, loss_epochs = parse_ablation_log(loss_log, 'Loss')
    cont_entries, cont_epochs = parse_ablation_log(cont_log, 'Contrastive') if os.path.exists(cont_log) else ([], [])

    all_entries = enc_entries + fus_entries + loss_entries + cont_entries
    all_epochs = enc_epochs + fus_epochs + loss_epochs + cont_epochs

    print(f"Total entries: {len(all_entries)} (Encoder: {len(enc_entries)}, Fusion: {len(fus_entries)}, Loss: {len(loss_entries)}, Contrastive: {len(cont_entries)})")

    # 1. Root Master Dashboard
    master_html = os.path.join(base_dir, 'index.html')
    generate_html_page(
        page_type='master',
        title='ARISE-V6 Ablation Master Suite (5-Category Benchmark)',
        subtitle='Consolidated systematic evaluation of Spatial Encoders (6 variants), Multi-Modal Fusion (7 variants), Loss Objectives (6 variants), and Contrastive Objectives (7 variants) organized across the 5 standard evaluation checkpoints.',
        entries=all_entries,
        epochs=all_epochs,
        output_file=master_html,
        root_rel=''
    )

    # 2. Encoder Ablation Dedicated Dashboard
    enc_html = os.path.join(base_dir, 'encoder-ablation', 'index.html')
    generate_html_page(
        page_type='encoder',
        title='ARISE-V6 Spatial Encoder Ablation Benchmark',
        subtitle='Systematic 5-category evaluation across 6 spatial graph encoder topologies on 6 spatial multi-omics benchmarks.',
        entries=enc_entries,
        epochs=enc_epochs,
        output_file=enc_html,
        root_rel='../'
    )

    # 3. Fusion Ablation Dedicated Dashboard
    fus_html = os.path.join(base_dir, 'fusion-ablation', 'index.html')
    generate_html_page(
        page_type='fusion',
        title='ARISE-V6 Fusion Ablation Benchmark',
        subtitle='Systematic 5-category evaluation across 7 cross-modal fusion strategies on 6 spatial multi-omics benchmarks.',
        entries=fus_entries,
        epochs=fus_epochs,
        output_file=fus_html,
        root_rel='../'
    )

    # 4. Loss Ablation Dedicated Dashboard
    if loss_entries:
        loss_html = os.path.join(base_dir, 'loss-ablation', 'index.html')
        generate_html_page(
            page_type='loss',
            title='ARISE-V6 Loss Objectives Ablation Benchmark',
            subtitle='Systematic 5-category evaluation across 6 loss objectives & regularization functions on 6 spatial multi-omics benchmarks.',
            entries=loss_entries,
            epochs=loss_epochs,
            output_file=loss_html,
            root_rel='../'
        )

    # 5. Contrastive Ablation Dedicated Dashboard (if log exists)
    if cont_entries:
        cont_html = os.path.join(base_dir, 'contrastive-ablation', 'index.html')
        generate_html_page(
            page_type='contrastive',
            title='ARISE-V6 Contrastive Objectives Ablation Benchmark',
            subtitle='Systematic 5-category evaluation across contrastive learning configurations on 6 spatial multi-omics benchmarks.',
            entries=cont_entries,
            epochs=cont_epochs,
            output_file=cont_html,
            root_rel='../'
        )

    print("\n✅ All 5-Category HTML Dashboards generated successfully!")

if __name__ == '__main__':
    main()
