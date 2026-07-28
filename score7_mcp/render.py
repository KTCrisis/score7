"""Rendu de la fiche markdown (format des analyses Renoise)."""

from __future__ import annotations


def render_markdown(r: dict) -> str:
    k, sp, st = r["key"], r["spectral"], r["stereo"]
    ld, dyn = r["loudness"], r["structure"]
    L = []
    L.append(f"# Analyse audio — {r['title']}\n")
    L.append(f"**Fichier :** `{r['file']}`  ")
    met, tp = r.get("meter") or {}, r.get("tempo") or {}
    meter_txt = ""
    if met.get("meter"):
        meter_txt = f" | **Métrique :** {met['meter']}"
        if met.get("meter_confidence") is not None:
            meter_txt += f" (conf. {met['meter_confidence']:.2f})"
        if met.get("source") == "madmom":
            meter_txt += " (madmom)"
    L.append(f"**Durée :** {dyn['duration_s']} s | **BPM estimé :** {r['bpm']}{meter_txt}  ")
    if tp.get("bpm_confidence") is not None and tp["bpm_confidence"] < 0.5:
        alts = ", ".join(f"{c['bpm']} ({c['strength']:.0%})" for c in tp["bpm_candidates"])
        L.append(f"*Tempo ambigu (octave) — candidats : {alts}.*  ")
    L.append("*Estimation statistique (chroma/spectral) — pas une transcription exacte.*\n")

    L.append("## Tonalité\n")
    detector = " *(CNN madmom)*" if k.get("source") == "madmom_cnn" else ""
    line = (f"**{k['root']} {k['mode']}** (score {k['score']}){detector} — "
            f"second candidat : {k['runner_up']}.")
    if k.get("corrected"):
        line += f" *Krumhansl seul disait {k['krumhansl_only']} ; corrigé par le vote d'accords.*"
    if k.get("changed_by_melody"):
        line += (f" *Mode tranché par la tonique de la mélodie : "
                 f"{k['before_melody']} → {k['root']} {k['mode']}.*")
    L.append(line + "\n")

    src = r.get("chords_source", "template")
    src_label = {"btc": "BTC — transformer, vocabulaire riche",
                 "madmom": "madmom — deep chroma + CRF",
                 "template": "template chroma (estimation cosinus)"}.get(src, src)
    L.append(f"## Grille d'accords — {src_label}\n")
    grid = r["chords"]
    if grid:
        rich = any(s.get("chord_full") for s in grid)
        if src == "template":
            L.append("| Temps (s) | Accord | Beats | Confiance |")
            L.append("|-----------|--------|-------|-----------|")
            for s in grid[:40]:
                t = f"{s['time']:.1f}" if s.get("time") is not None else "-"
                L.append(f"| {t} | **{s['chord']}** | {s['beats']} | {s['conf']:.2f} |")
        else:
            L.append("| Temps (s) | Accord | Beats |")
            L.append("|-----------|--------|-------|")
            for s in grid[:40]:
                t = f"{s['time']:.1f}" if s.get("time") is not None else "-"
                L.append(f"| {t} | **{s.get('chord_full') or s['chord']}** | {s['beats']} |")
        chain_key = "chord_full" if rich else "chord"
        L.append(f"\n**Enchaînement :** "
                 f"{' → '.join((s.get(chain_key) or s['chord']) for s in grid[:16])}\n")
        if src == "template":
            L.append("*Faible confiance / enchaînement instable = harmonie suspendue (ambient) — "
                     "à corriger à l'oreille.*\n")
        else:
            L.append("*Reconnaissance par modèle entraîné — fiable sur maj/min, les qualités "
                     "fines (7e, sus, dim) restent à confirmer à l'oreille.*\n")
    else:
        L.append("*Pas de grille exploitable.*\n")

    L.append("## Structure dynamique\n")
    L.append("| Section | t (s) | RMS (dB) |")
    L.append("|---------|-------|----------|")
    for i, c in enumerate(dyn["coarse"], 1):
        L.append(f"| {i} | {c['t_start']}–{c['t_end']} | {c['rms_db']} |")
    if dyn["layer_onsets"]:
        L.append(f"\n**Paliers d'entrée détectés à :** "
                 f"{', '.join(str(x) + 's' for x in dyn['layer_onsets'])}\n")

    L.append("## Profil spectral\n")
    L.append(f"- Centroïde : **{sp['centroid_hz']} Hz** — {sp['description']}")
    L.append(f"- Rolloff 85 % : {sp['rolloff85_hz']} Hz | Bande passante : {sp['bandwidth_hz']} Hz")
    L.append(f"- Flatness : {sp['flatness']} (0 = tonal, 1 = bruité)\n")

    L.append("## Stéréo\n")
    L.append(f"- Corrélation L/R : {st['correlation']} | ratio side/mid : {st['side_mid_ratio']}")
    L.append(f"- {st['description']}\n")

    L.append("## Loudness & dynamique\n")
    L.append(f"- LUFS intégré : **{ld['integrated_lufs']}** | Peak : {ld['peak_dbfs']} dBFS | "
             f"RMS : {ld['rms_dbfs']} dBFS")
    L.append(f"- Crest factor : {ld['crest_factor_db']} dB (élevé = dynamique préservée)\n")

    if r.get("melody"):
        m = r["melody"]
        L.append("## Mélodie extraite\n")
        L.append(f"Méthode : {m['method']} — {m['n_notes']} notes → `{m['midi']}`\n")
        if m.get("poly_probe") == "indisponible":
            L.append("*Sonde de polyphonie indisponible : la route n'a pas été mesurée, "
                     "le matériau est traité comme monophonique quel qu'il soit.*\n")
        elif m.get("polyphony") is not None:
            L.append(f"*Polyphonie mesurée : {m['polyphony']} (sonde {m['poly_probe']}).*\n")
        L.append(f"**Classes de hauteur :** {m['pitch_classes']}\n")
        L.append(f"**Séquence :** {m['sequence']}\n")
        L.append("*Ligne supérieure estimée — à nettoyer dans Renoise/MuseScore.*\n")
    rp = r.get("rhythm_pattern")
    if rp and rp.get("available"):
        L.append("## Rythme — pattern batterie\n")
        L.append(f"Grille repliée sur une mesure, {rp['steps_per_beat']} pas par temps "
                 f"(`X` fort · `x` faible · `.` silence) :\n")
        L.append("```")
        for band in ("kick", "snare", "hats"):
            b = rp["bands"].get(band)
            if b:
                L.append(f"{band:5s} | {b['grid']}   ({b['density_hz']} onsets/s)")
        L.append("```")
        groove = []
        if rp.get("microtiming_ms") is not None:
            groove.append(f"microtiming moyen {rp['microtiming_ms']} ms")
        if rp.get("swing_ratio") is not None:
            tag = ("droit" if rp["swing_pct"] < 20 else
                   "swing marqué" if rp["swing_pct"] > 60 else "légèrement swingué")
            groove.append(f"swing {rp['swing_ratio']} ({rp['swing_pct']} % — {tag})")
        if groove:
            L.append(f"\n**Groove :** {', '.join(groove)}.")
        L.append(f"\n*{rp['note']}.*\n")

    tex = r.get("stem_textures")
    if tex:
        L.append("## Texture par stem\n")
        L.append("| Stem | Part | Caractère | Centroïde | Flux | Percussif | Stéréo |")
        L.append("|------|------|-----------|-----------|------|-----------|--------|")
        for name in ("drums", "bass", "other", "vocals"):
            t = tex.get(name)
            if not t or "error" in t:
                continue
            share = f"{t['energy_share']:.0%}" if t.get("energy_share") is not None else "-"
            L.append(f"| {name} | {share} | {t['description']} | "
                     f"{t['spectral']['centroid_hz']} Hz | {t['spectral_flux']} | "
                     f"{t['percussive_ratio']} | {t['stereo']['side_mid_ratio']} |")
        L.append("\n*Part = énergie RMS relative dans le mix. Flux élevé = timbre mouvant ; "
                 "percussif proche de 1 = transitoire, proche de 0 = soutenu.*\n")

    if r.get("stems"):
        L.append(f"## Stems séparés\n\n`{r['stems']}`\n")

    return "\n".join(L)
