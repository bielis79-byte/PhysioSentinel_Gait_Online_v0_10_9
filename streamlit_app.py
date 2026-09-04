import json
import math
import os
import io
import zipfile
import tempfile
import re
import shutil
import tempfile
import traceback
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import requests
import streamlit as st
import toml
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def _safe_filename(value):
    s = str(value or "").strip()
    if not s:
        return "sin_nombre"
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:80]


def _metrics_dataframe(metrics):
    rows = []
    for m in metrics or []:
        rows.append({
            "key": m.get("key"),
            "label": m.get("label"),
            "value": m.get("value"),
            "unit": m.get("unit"),
            "quality": m.get("quality"),
            "notes": m.get("notes"),
        })
    return pd.DataFrame(rows)


def _plot_column_png(chart_df, column, title=None):
    """
    Genera un PNG en memoria. No escribe nada en Supabase ni en disco persistente.
    """
    if chart_df is None or column not in chart_df.columns:
        return None
    y = pd.to_numeric(chart_df[column], errors="coerce")
    if "time_s" in chart_df.columns:
        x = pd.to_numeric(chart_df["time_s"], errors="coerce")
        xlabel = "Tiempo (s)"
    else:
        x = np.arange(len(chart_df))
        xlabel = "Muestra"

    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(np.asarray(x)[valid], np.asarray(y)[valid], linewidth=1.6)
    ax.set_title(title or str(column))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(str(column))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()


def _add_chart_bundle_to_zip(zf, chart_df, folder_name):
    if chart_df is None or not isinstance(chart_df, pd.DataFrame) or chart_df.empty:
        return 0

    # Datos fuente del gráfico
    zf.writestr(
        f"{folder_name}/datos_graficos.csv",
        chart_df.to_csv(index=False).encode("utf-8-sig")
    )

    count = 0
    ignore = {"frame", "time_s"}
    for col in chart_df.columns:
        if col in ignore:
            continue
        try:
            png = _plot_column_png(chart_df, col, title=col)
            if png:
                zf.writestr(
                    f"{folder_name}/graficos/{count+1:02d}_{_safe_filename(col)}.png",
                    png
                )
                count += 1
        except Exception:
            continue
    return count


def build_export_zip(
    metrics,
    chart=None,
    chart_front=None,
    chart_lateral=None,
    technical_report="",
    patient_report="",
    annotated_video_bytes=None,
    annotated_video2_bytes=None,
    patient_code="",
    record_name="",
    patient_age=0,
    patient_sex="No especificado",
    record_date=None,
    app_version="",
):
    """
    Crea un paquete ZIP totalmente en memoria.
    No persiste gráficos, vídeos ni resultados pesados en Supabase.
    """
    out = io.BytesIO()

    meta = {
        "app_version": app_version,
        "patient_code": patient_code,
        "record_name": record_name,
        "patient_age": int(patient_age or 0),
        "patient_sex": patient_sex,
        "record_date": record_date.isoformat() if hasattr(record_date, "isoformat") else str(record_date or ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "00_metadatos.json",
            json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        )

        mdf = _metrics_dataframe(metrics)
        if not mdf.empty:
            zf.writestr(
                "01_resultados/resultados_metricas.csv",
                mdf.to_csv(index=False).encode("utf-8-sig")
            )
            zf.writestr(
                "01_resultados/resultados_metricas.json",
                json.dumps(mdf.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str).encode("utf-8")
            )

        if technical_report:
            zf.writestr("02_informes/informe_tecnico.txt", technical_report.encode("utf-8"))
        if patient_report:
            zf.writestr("02_informes/informe_paciente.txt", patient_report.encode("utf-8"))

        graph_count = 0
        graph_count += _add_chart_bundle_to_zip(zf, chart, "03_graficos/vista_unica")
        graph_count += _add_chart_bundle_to_zip(zf, chart_front, "03_graficos/frontal")
        graph_count += _add_chart_bundle_to_zip(zf, chart_lateral, "03_graficos/lateral")

        if annotated_video_bytes:
            zf.writestr("04_video_anotado/video_anotado_cam01.mp4", annotated_video_bytes)
        if annotated_video2_bytes:
            zf.writestr("04_video_anotado/video_anotado_cam02.mp4", annotated_video2_bytes)

        # v0.10.0 · exportación del analizador del ciclo si fue generado
        try:
            if st.session_state.get("cycle_kinematic_png"):
                zf.writestr("05_ciclo_marcha/curva_cinematica_normalizada.png", st.session_state["cycle_kinematic_png"])
            if st.session_state.get("cycle_phase_png"):
                zf.writestr("05_ciclo_marcha/fases_por_extremidad.png", st.session_state["cycle_phase_png"])
            phase_table = st.session_state.get("cycle_phase_table")
            if isinstance(phase_table, pd.DataFrame) and not phase_table.empty:
                zf.writestr("05_ciclo_marcha/resumen_fases.csv", phase_table.to_csv(index=False).encode("utf-8-sig"))
        except Exception:
            pass

        manifest = (
            f"PhysioSentinel Gait {app_version}\n"
            f"Paciente/código: {patient_code}\n"
            f"Registro: {record_name}\n"
            f"Gráficos PNG exportados: {graph_count}\n\n"
            "Este paquete se genera bajo demanda en memoria y no implica "
            "almacenamiento permanente de gráficos o vídeos en Supabase.\n"
        )
        zf.writestr("LEEME_EXPORTACION.txt", manifest.encode("utf-8"))

    out.seek(0)
    return out.getvalue()





def _valid_dataframe(obj):
    return isinstance(obj, pd.DataFrame) and not obj.empty



def _support_summary_positions_to_video_frames(summary, seg):
    """
    v0.10.4 · _support_cycle_summary trabaja sobre posiciones 0..N-1 de la
    máscara. Para sincronizar con vídeo y curvas hay que convertir IC/TO/nextIC
    a los números de frame reales contenidos en seg["frame"].
    """
    if not isinstance(summary, dict) or not _valid_dataframe(seg):
        return summary
    frame_values = pd.to_numeric(seg["frame"], errors="coerce").to_numpy()
    n = len(frame_values)
    out = dict(summary)
    for key in ("ic", "to", "next_ic", "starts", "ends"):
        arr = np.asarray(summary.get(key, []), dtype=int)
        mapped = []
        for p in arr:
            if n == 0:
                continue
            # `ends` puede ser exclusivo == n; mapearlo al último frame + 1
            if key == "ends" and p >= n:
                if np.isfinite(frame_values[-1]):
                    mapped.append(int(frame_values[-1]) + 1)
                continue
            q = int(np.clip(p, 0, n-1))
            if np.isfinite(frame_values[q]):
                mapped.append(int(frame_values[q]))
        out[key] = np.asarray(mapped, dtype=int)
    return out


def _pair_cycles(left_cycles, right_cycles):
    """
    Empareja cada ciclo izquierdo con el derecho temporalmente más cercano.
    Evita mostrar por defecto ciclos que pertenecen a instantes muy diferentes.
    """
    pairs = []
    used_r = set()
    for li, lc in enumerate(left_cycles or []):
        lmid = (lc["ic_frame"] + lc["next_ic_frame"]) / 2.0
        best = None
        for ri, rc in enumerate(right_cycles or []):
            rmid = (rc["ic_frame"] + rc["next_ic_frame"]) / 2.0
            overlap = max(0, min(lc["next_ic_frame"], rc["next_ic_frame"]) - max(lc["ic_frame"], rc["ic_frame"]))
            dist = abs(lmid-rmid)
            score = dist - 1.5*overlap
            if best is None or score < best[0]:
                best = (score, ri)
        pairs.append((li, best[1] if best else None))

    if not pairs and right_cycles:
        for ri in range(len(right_cycles)):
            pairs.append((None, ri))
    return pairs


def _make_cycle_snapshot(seg, fps):
    """v0.10.9 · snapshot con cadena temporal validada."""
    if seg is None or not isinstance(seg,pd.DataFrame) or seg.empty:
        return None
    snap_seg=seg.copy().reset_index(drop=True)
    lmask,_,_=_support_mask_2d(snap_seg,"L",float(fps))
    rmask,_,_=_support_mask_2d(snap_seg,"R",float(fps))
    L0=_support_cycle_summary(lmask,float(fps))
    R0=_support_cycle_summary(rmask,float(fps))
    Lp,Rp,event_source,event_info=_select_temporal_event_chain(snap_seg,float(fps),L0,R0)
    L=_support_summary_positions_to_video_frames(Lp,snap_seg)
    R=_support_summary_positions_to_video_frames(Rp,snap_seg)
    return {
        "seg":snap_seg,"fps":float(fps),"left":L,"right":R,
        "event_source":event_source,
        "event_quality":event_info.get("quality","") if isinstance(event_info,dict) else "",
        "event_notes":event_info.get("reason","") if isinstance(event_info,dict) else "",
    }


def _video_frame_from_bytes(video_bytes, frame_no):
    """
    Extrae un frame de un MP4 guardado en session_state sin persistirlo.
    Usa un archivo temporal efímero solo durante la lectura.
    """
    if not video_bytes:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return frame
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _encode_phase_overlay(frame, frame_no, left_cycle, right_cycle):
    if frame is None:
        return None
    lines = []
    for cyc, label in [(left_cycle, "IZQ"), (right_cycle, "DER")]:
        info = _cycle_phase_for_frame(cyc, int(frame_no)) if cyc else None
        if info:
            pct, phase = info
            lines.append(f"{label}: {phase} · {pct:.1f}%")
    if not lines:
        lines = ["Fuera del ciclo seleccionado"]

    y = 35
    for line in lines:
        cv2.putText(frame, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2, cv2.LINE_AA)
        y += 32

    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),92])
    return enc.tobytes() if ok else None


def _phase_name_from_pct(pct, stance_pct=60.0):
    """
    Conventional descriptive subdivision of the gait cycle.
    Percentages are contextual, not force-plate validated.
    """
    if not np.isfinite(pct):
        return "No calculable"
    pct = float(pct) % 100.0
    stance_pct = float(np.clip(stance_pct if np.isfinite(stance_pct) else 60.0, 45.0, 80.0))

    # Split stance into loading / mid / terminal / pre-swing proportionally.
    loading_end = min(12.0, stance_pct * 0.20)
    mid_end = max(loading_end + 8.0, stance_pct * 0.55)
    terminal_end = max(mid_end + 8.0, stance_pct * 0.85)
    if pct < loading_end:
        return "Respuesta a la carga"
    if pct < mid_end:
        return "Apoyo medio"
    if pct < terminal_end:
        return "Apoyo terminal"
    if pct < stance_pct:
        return "Pre-oscilación / propulsión"

    swing_span = max(1.0, 100.0 - stance_pct)
    swing_rel = (pct - stance_pct) / swing_span
    if swing_rel < 0.33:
        return "Swing inicial"
    if swing_rel < 0.67:
        return "Swing medio"
    return "Swing terminal"



def _summary_from_positions(ic_positions, to_positions, next_ic_positions, fps, source="robust_distal"):
    ic=np.asarray(ic_positions,dtype=int); to=np.asarray(to_positions,dtype=int); nxt=np.asarray(next_ic_positions,dtype=int)
    n=min(len(ic),len(to),len(nxt)); ic,to,nxt=ic[:n],to[:n],nxt[:n]
    keep=(nxt>ic)&(to>ic)&(to<nxt); ic,to,nxt=ic[keep],to[keep],nxt[keep]
    cycle=(nxt-ic)/float(fps) if len(ic) else np.asarray([],float)
    stance=(to-ic)/float(fps) if len(ic) else np.asarray([],float)
    swing=(nxt-to)/float(fps) if len(ic) else np.asarray([],float)
    stance_pct=np.where(cycle>0,stance/cycle*100.0,np.nan) if len(ic) else np.asarray([],float)
    score=90.0 if len(cycle)>=3 else (68.0 if len(cycle)==2 else 40.0)
    quality="Alta" if score>=75 else ("Moderada" if score>=55 else "No fiable")
    return {
        "stance":stance,"swing":swing,"cycle":cycle,"stance_pct":stance_pct,
        "swing_pct":100.0-stance_pct if len(stance_pct) else np.asarray([],float),
        "ic":ic,"to":to,"next_ic":nxt,"starts":ic.copy(),"ends":to.copy(),
        "quality":quality,"score":float(score),"cycle_error_pct":np.nan,
        "fragmentation_index":np.nan,"reason":"cadena robusta distal",
        "event_source":source,"coverage_mask":np.zeros(0,dtype=bool),
    }


def _robust_distal_support_summaries(seg, fps):
    """v0.10.9 · IC/TO alternantes con QC temporal amplio."""
    empty=_summary_from_positions([],[],[],fps)
    if seg is None or len(seg)<max(45,int(round(2.5*fps))):
        return empty,empty,{"quality":"No fiable","reason":"segmento insuficiente"}

    need=["LAnkle_y","LHeel_y","LBigToe_y","RAnkle_y","RHeel_y","RBigToe_y",
          "LHip_y","RHip_y","LShoulder_y","RShoulder_y"]
    if not all(c in seg.columns for c in need):
        return empty,empty,{"quality":"No calculable","reason":"faltan trayectorias distales"}

    def a(c): return pd.to_numeric(seg[c],errors="coerce").to_numpy(float)
    ly=(a("LAnkle_y")+a("LHeel_y")+a("LBigToe_y"))/3.0
    ry=(a("RAnkle_y")+a("RHeel_y")+a("RBigToe_y"))/3.0
    torso=np.abs((a("LHip_y")+a("RHip_y"))/2.0-(a("LShoulder_y")+a("RShoulder_y"))/2.0)

    def clean(x):
        s=pd.Series(x).replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both")
        return s.to_numpy(float) if s.notna().sum()>=max(20,int(0.7*len(s))) else None
    ly,ry,torso=clean(ly),clean(ry),clean(torso)
    if ly is None or ry is None or torso is None:
        return empty,empty,{"quality":"No calculable","reason":"tracking distal insuficiente"}

    floor=np.nanpercentile(torso[np.isfinite(torso)],20) if np.isfinite(torso).any() else 1.0
    sig=(ry-ly)/np.maximum(torso,max(floor,1e-6))
    ws=max(5,int(round(0.18*fps))|1); wl=max(ws+2,int(round(1.8*fps))|1)
    sig=pd.Series(sig).rolling(ws,center=True,min_periods=1).median().to_numpy(float)
    sig=sig-pd.Series(sig).rolling(wl,center=True,min_periods=1).mean().to_numpy(float)

    amp=np.nanmedian(np.abs(sig-np.nanmedian(sig)))*1.4826
    sd=np.nanstd(sig)
    if not np.isfinite(amp) or amp<=1e-8: amp=sd
    if not np.isfinite(amp) or amp<=1e-8:
        return empty,empty,{"quality":"No calculable","reason":"sin periodicidad distal"}

    min_dist=max(2,int(round(0.28*fps))); prom=max(0.42*amp,0.28*sd)
    pos,_=find_peaks(sig,distance=min_dist,prominence=prom)
    neg,_=find_peaks(-sig,distance=min_dist,prominence=prom)
    cand=[(int(i),"R",abs(float(sig[i]))) for i in pos]+[(int(i),"L",abs(float(sig[i]))) for i in neg]
    cand.sort(key=lambda z:z[0])

    collapsed=[]
    for ev in cand:
        if collapsed and collapsed[-1][1]==ev[1]:
            if ev[2]>collapsed[-1][2]: collapsed[-1]=ev
        else:
            collapsed.append(ev)

    alt=[]
    for ev in collapsed:
        if not alt:
            alt.append(ev); continue
        dt=(ev[0]-alt[-1][0])/float(fps)
        if dt<0.30:
            if ev[2]>alt[-1][2]: alt[-1]=ev
            continue
        if ev[1]==alt[-1][1]:
            if ev[2]>alt[-1][2]: alt[-1]=ev
            continue
        alt.append(ev)

    if len(alt)<5:
        return empty,empty,{"quality":"No fiable","reason":f"solo {len(alt)} eventos alternantes"}

    dts=np.diff([e[0] for e in alt])/float(fps)
    finite=dts[np.isfinite(dts)&(dts>=0.30)&(dts<=1.80)]
    if len(finite)<3:
        return empty,empty,{"quality":"No fiable","reason":"intervalos insuficientes"}
    med=float(np.nanmedian(finite)); mad=float(np.nanmedian(np.abs(finite-med)))
    lo=max(0.30,med-max(0.18,3.5*mad)); hi=min(1.80,med+max(0.25,3.5*mad))
    cleaned=[alt[0]]
    for ev in alt[1:]:
        dt=(ev[0]-cleaned[-1][0])/float(fps)
        if lo<=dt<=hi and ev[1]!=cleaned[-1][1]:
            cleaned.append(ev)
    if len(cleaned)<5: cleaned=alt

    side_events={"L":[e[0] for e in cleaned if e[1]=="L"],"R":[e[0] for e in cleaned if e[1]=="R"]}
    foot={"L":ly,"R":ry}

    def build(side):
        ics=side_events[side]; out_ic=[]; out_to=[]; out_next=[]; y=foot[side]
        for x0,x1 in zip(ics[:-1],ics[1:]):
            cyc=(x1-x0)/float(fps)
            if not (0.65<=cyc<=3.20): continue
            p0=max(x0+1,min(int(round(x0+0.45*(x1-x0))),x1-2))
            p1=max(p0+1,min(int(round(x0+0.80*(x1-x0))),x1-1))
            yy=y[p0:p1+1]
            if len(yy)<3 or not np.isfinite(yy).any(): continue
            ys=pd.Series(yy).rolling(max(3,int(round(0.10*fps))|1),center=True,min_periods=1).mean().to_numpy(float)
            to=p0+int(np.nanargmin(np.gradient(ys)))
            pct=(to-x0)/(x1-x0)*100.0
            if 45.0<=pct<=80.0:
                out_ic.append(x0); out_to.append(to); out_next.append(x1)
        return _summary_from_positions(out_ic,out_to,out_next,fps)

    L,R=build("L"),build("R")
    ncy=len(L.get("cycle",[]))+len(R.get("cycle",[]))
    q="Alta" if ncy>=6 else ("Moderada" if ncy>=4 else "No fiable")
    return L,R,{"quality":q,"reason":f"{len(cleaned)} eventos alternantes; {ncy} ciclos robustos"}


def _summary_temporally_suspicious(summary):
    if not isinstance(summary,dict): return True
    cyc=np.asarray(summary.get("cycle",[]),float); pct=np.asarray(summary.get("stance_pct",[]),float)
    if len(cyc)<2 or len(pct)<2: return True
    good=np.isfinite(cyc)&np.isfinite(pct)&(cyc>=0.65)&(cyc<=3.20)&(pct>=45.0)&(pct<=80.0)
    return int(good.sum())<2


def _select_temporal_event_chain(seg,fps,L0,R0):
    Lr,Rr,info=_robust_distal_support_summaries(seg,fps)
    original_bad=_summary_temporally_suspicious(L0) or _summary_temporally_suspicious(R0)
    robust_good=(not _summary_temporally_suspicious(Lr)) and (not _summary_temporally_suspicious(Rr))
    if original_bad and robust_good:
        return Lr,Rr,"robust_distal",info
    return L0,R0,"support_mask",info


def _build_side_cycles(summary, fps, side):
    """
    v0.10.1 · Convierte el resumen IC→TO→nextIC en ciclos explícitos.
    Usa directamente los tres arrays validados por _support_cycle_summary,
    evitando asumir que ic contiene también el IC siguiente.
    """
    rows = []
    ics = np.asarray(summary.get("ic", []), dtype=int)
    tos = np.asarray(summary.get("to", []), dtype=int)
    next_ics = np.asarray(summary.get("next_ic", []), dtype=int)
    cycles = np.asarray(summary.get("cycle", []), dtype=float)
    stances = np.asarray(summary.get("stance", []), dtype=float)

    n = min(len(ics), len(tos), len(next_ics), len(cycles), len(stances))
    for i in range(n):
        ic = int(ics[i])
        to = int(tos[i])
        next_ic = int(next_ics[i])
        if not (ic < to <= next_ic):
            continue

        cycle_s = float((next_ic - ic) / fps) if next_ic > ic else np.nan
        stance_s = float((to - ic) / fps)
        swing_s = float((next_ic - to) / fps)
        stance_pct = float(stance_s / cycle_s * 100.0) if np.isfinite(cycle_s) and cycle_s > 0 else np.nan

        rows.append({
            "side": side,
            "cycle_index": len(rows) + 1,
            "ic_frame": ic,
            "to_frame": to,
            "next_ic_frame": next_ic,
            "cycle_s": cycle_s,
            "stance_s": stance_s,
            "swing_s": swing_s,
            "stance_pct": stance_pct,
        })
    return rows


def _cycle_phase_for_frame(cycle, frame):
    if not cycle:
        return None
    ic = cycle["ic_frame"]
    nic = cycle["next_ic_frame"]
    if frame < ic or frame > nic or nic <= ic:
        return None
    pct = (frame - ic) / (nic - ic) * 100.0
    phase = _phase_name_from_pct(pct, cycle.get("stance_pct", 60.0))
    return float(pct), phase


def _find_cycle_for_frame(cycles, frame):
    for c in cycles:
        if c["ic_frame"] <= frame <= c["next_ic_frame"]:
            return c
    return None


def _normalized_cycle_series(seg, cycle, columns):
    """
    v0.10.4 · Extrae el ciclo usando frames REALES y devuelve señal 0–100%.
    Si hay frames perdidos por tracking, interpola solo para visualización.
    """
    if cycle is None or seg is None or seg.empty:
        return pd.DataFrame()
    ic, nic = int(cycle["ic_frame"]), int(cycle["next_ic_frame"])
    if nic <= ic:
        return pd.DataFrame()

    part = seg[(pd.to_numeric(seg["frame"],errors="coerce") >= ic) &
               (pd.to_numeric(seg["frame"],errors="coerce") <= nic)].copy()
    if part.empty:
        return pd.DataFrame()

    part["cycle_pct"] = (
        (pd.to_numeric(part["frame"],errors="coerce") - ic) / float(nic-ic) * 100.0
    )
    keep=["frame","cycle_pct"]+[c for c in columns if c in part.columns]
    part=part[keep].sort_values("cycle_pct")

    for c in columns:
        if c in part.columns:
            part[c]=pd.to_numeric(part[c],errors="coerce")
            if part[c].notna().sum() >= 2:
                part[c]=part[c].interpolate(limit_direction="both")
    return part


def _compute_display_kinematics(seg):
    """
    v0.10.4 · Señales para el visor del ciclo.

    Prioriza las columnas cinemáticas YA calculadas por compute_metrics para
    evitar que la pestaña 9 reconstruya una señal distinta. Solo calcula
    aliases/fallbacks cuando una variable todavía no existe.
    """
    out = seg.copy()

    aliases = {
        "L_knee_flex": "Flexión rodilla Izquierda (°)",
        "R_knee_flex": "Flexión rodilla Derecha (°)",
        "L_hip_flex": "Flexión cadera Izquierda (°)",
        "R_hip_flex": "Flexión cadera Derecha (°)",
        "L_ankle_angle": "Ángulo tobillo Izquierda (°)",
        "R_ankle_angle": "Ángulo tobillo Derecha (°)",
        "pelvis_obliquity": "Oblicuidad pélvica (°)",
        "shoulder_obliquity": "Oblicuidad hombros (°)",
        "trunk_lateral_lean": "Inclinación tronco (°)",
        "L_frontal_knee_dev": "Desviación frontal rodilla Izquierda (°)",
        "R_frontal_knee_dev": "Desviación frontal rodilla Derecha (°)",
        "L_foot_progress_proj": "Orientación pie Izquierda (°)",
        "R_foot_progress_proj": "Orientación pie Derecha (°)",
        "L_rearfoot_tilt_proj": "Retropié Izquierda (°)",
        "R_rearfoot_tilt_proj": "Retropié Derecha (°)",
    }
    for src, label in aliases.items():
        if src in out.columns:
            out[label] = pd.to_numeric(out[src], errors="coerce")

    # Fallback rodilla si por cualquier motivo no existe la columna derivada.
    def xy(name):
        return out[f"{name}_x"].to_numpy(float), out[f"{name}_y"].to_numpy(float)

    for side_label, side, hip, knee, ankle in [
        ("Izquierda","L","LHip","LKnee","LAnkle"),
        ("Derecha","R","RHip","RKnee","RAnkle"),
    ]:
        label = f"Flexión rodilla {side_label} (°)"
        if label not in out.columns:
            need=[f"{p}_{a}" for p in (hip,knee,ankle) for a in ("x","y")]
            if all(c in out.columns for c in need):
                hx,hy=xy(hip); kx,ky=xy(knee); ax,ay=xy(ankle)
                v1=np.c_[hx-kx,hy-ky]; v2=np.c_[ax-kx,ay-ky]
                den=np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)
                dot=np.sum(v1*v2,axis=1)
                cosang=np.divide(dot,den,out=np.full(len(out),np.nan),where=den>1e-9)
                out[label]=180.0-np.degrees(np.arccos(np.clip(cosang,-1,1)))

    # Suavizado ligero SOLO para presentación; no cambia las métricas.
    display_cols=[c for c in out.columns if any(t in c for t in (
        "Flexión ","Ángulo tobillo","Oblicuidad ","Inclinación tronco",
        "Desviación frontal","Orientación pie","Retropié"
    ))]
    for c in display_cols:
        s=pd.to_numeric(out[c],errors="coerce")
        if s.notna().sum() >= 3:
            out[c]=s.interpolate(limit=3,limit_direction="both").rolling(3,center=True,min_periods=1).mean()

    return out


def _phase_segments_for_stance(stance_pct):
    stance = float(np.clip(stance_pct if np.isfinite(stance_pct) else 60.0, 35.0, 90.0))
    loading = min(12.0, 0.20*stance)
    mid = max(loading+5.0, 0.55*stance)
    terminal = max(mid+5.0, 0.85*stance)
    swing = 100.0-stance
    s1 = stance + swing/3.0
    s2 = stance + 2.0*swing/3.0
    return [
        (0, loading, "Carga"),
        (loading, mid, "Apoyo medio"),
        (mid, terminal, "Apoyo terminal"),
        (terminal, stance, "Propulsión"),
        (stance, s1, "Swing inicial"),
        (s1, s2, "Swing medio"),
        (s2, 100, "Swing terminal"),
    ]





def _macro_phase(label):
    """
    Simplifica la visualización global a tres bloques clínicamente legibles.
    El detalle fino sigue mostrándose en los indicadores de fase actual.
    """
    if label in ("Respuesta a la carga", "Carga", "Apoyo medio", "Apoyo terminal"):
        return "Apoyo"
    if label in ("Pre-oscilación / propulsión", "Propulsión"):
        return "Propulsión"
    if label in ("Swing inicial", "Swing medio", "Swing terminal"):
        return "Swing"
    return "Otro"


def _whole_segment_kinematic_figure(kin, variable_left, variable_right,
                                    fmin, fps, current_frame):
    """
    v0.10.8 · Curvas sobre el MISMO eje temporal (segundos) del mapa de fases.
    Esto elimina la falsa impresión de desalineación causada por comparar:
      mapa = tiempo total
      curva = 0–100% de un solo ciclo.
    """
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    plotted = False

    x = (
        pd.to_numeric(kin["frame"], errors="coerce").to_numpy(float) - float(fmin)
    ) / float(fps)

    def add(var, label):
        nonlocal plotted
        if var not in kin.columns:
            return
        y = pd.to_numeric(kin[var], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if int(ok.sum()) >= 2:
            ax.plot(x[ok], y[ok], linewidth=1.5, label=label)
            plotted = True

    add(variable_left, f"Izq · {variable_left}")
    add(variable_right, f"Der · {variable_right}")

    tcur = (float(current_frame)-float(fmin))/float(fps)
    ax.axvline(tcur, linewidth=2.0, label=f"Cursor {tcur:.2f} s")
    ax.set_xlabel("Tiempo del segmento (s)", fontsize=8)
    ax.set_ylabel("Magnitud 2D", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Cinemática temporal · alineada con el mapa de fases", fontsize=10)
    ax.grid(True, alpha=0.20)
    if plotted:
        ax.legend(loc="best", fontsize=6.5, ncol=2)
    else:
        ax.text(
            0.5, 0.5, "Sin datos suficientes para las variables seleccionadas",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.5
        )
    fig.tight_layout(pad=0.6)
    return fig


def _draw_phase_progress_overlay(frame, frame_no, left_cycles, right_cycles):
    """
    Overlay compacto para el vídeo reproducible:
    fase detallada + progreso de ciclo de cada extremidad.
    """
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    entries = []
    for side, cycles in [("IZQ", left_cycles), ("DER", right_cycles)]:
        cyc = _find_cycle_for_frame(cycles, frame_no) if cycles else None
        info = _cycle_phase_for_frame(cyc, frame_no) if cyc else None
        if info:
            pct, phase = info
            entries.append((side, phase, pct))
        else:
            entries.append((side, "Entre ciclos", np.nan))

    # Fondo semitransparente superior.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,78), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    y = 26
    for side, phase, pct in entries:
        txt = f"{side}: {phase}" + (f" · {pct:.0f}%" if np.isfinite(pct) else "")
        cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255,255,255), 2, cv2.LINE_AA)
        if np.isfinite(pct):
            x0, x1 = 12, min(w-12, 240)
            yy = y + 9
            cv2.rectangle(frame, (x0,yy), (x1,yy+5), (100,100,100), -1)
            xf = int(round(x0 + (x1-x0)*float(np.clip(pct,0,100))/100.0))
            cv2.rectangle(frame, (x0,yy), (xf,yy+5), (255,255,255), -1)
        y += 35
    return frame


def _build_phase_review_video_bytes(source_bytes, left_cycles, right_cycles, fps):
    """
    Genera una vez por sesión un MP4 de revisión con fases/progreso embebidos.
    No se guarda en Supabase.
    """
    if not source_bytes:
        return None

    src_path = raw_path = out_path = None
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="gait_phase_review_"))
        src_path = tmpdir / "source.mp4"
        raw_path = tmpdir / "phase_raw.mp4"
        out_path = tmpdir / "phase_review.mp4"
        src_path.write_bytes(source_bytes)

        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            return None
        fps_in = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in,
            (width, height),
        )
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _draw_phase_progress_overlay(
                frame, frame_no, left_cycles, right_cycles
            )
            writer.write(frame)
            frame_no += 1
        cap.release()
        writer.release()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = (
            f'"{ffmpeg}" -y -loglevel error -i "{raw_path}" '
            f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out_path}"'
        )
        rc = os.system(cmd)
        if rc == 0 and out_path.exists():
            return out_path.read_bytes()
        return raw_path.read_bytes() if raw_path.exists() else None
    finally:
        try:
            if 'tmpdir' in locals() and tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _change_video_speed_bytes(video_bytes, speed):
    """
    Cambia velocidad de reproducción manteniendo todos los frames.
    speed=0.5 -> cámara lenta 50%; speed=2 -> doble velocidad.
    """
    if not video_bytes:
        return None
    speed = float(speed)
    if speed <= 0:
        speed = 1.0
    if abs(speed-1.0) < 1e-9:
        return video_bytes

    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="gait_speed_"))
        src = tmpdir / "src.mp4"
        out = tmpdir / "out.mp4"
        src.write_bytes(video_bytes)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        # setpts = PTS/speed
        cmd = (
            f'"{ffmpeg}" -y -loglevel error -i "{src}" '
            f'-filter:v "setpts=PTS/{speed:.6f}" '
            f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out}"'
        )
        rc = os.system(cmd)
        return out.read_bytes() if rc == 0 and out.exists() else None
    finally:
        try:
            if 'tmpdir' in locals() and tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _phase_intervals_for_cycle(cycle, fps, fmin):
    """
    Convierte un ciclo IC→TO→IC en intervalos de fase sobre TIEMPO REAL del vídeo.
    Devuelve (t0_s, t1_s, label).
    """
    if not cycle or cycle["next_ic_frame"] <= cycle["ic_frame"]:
        return []

    ic = float(cycle["ic_frame"])
    nic = float(cycle["next_ic_frame"])
    stance_pct = float(cycle.get("stance_pct", 60.0))
    out = []
    for a, b, label in _phase_segments_for_stance(stance_pct):
        fa = ic + (a/100.0)*(nic-ic)
        fb = ic + (b/100.0)*(nic-ic)
        out.append(((fa-fmin)/fps, (fb-fmin)/fps, label))
    return out


def _whole_segment_phase_timeline(left_cycles, right_cycles, fps, fmin, fmax, current_frame):
    """
    v0.10.8 · Mapa temporal limpio.
    Solo tres categorías globales: Apoyo, Propulsión y Swing.
    El detalle fino se muestra arriba como fase actual.
    """
    duration = max(0.001, (float(fmax)-float(fmin))/float(fps))
    fig, ax = plt.subplots(figsize=(7.2, 1.75))
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.15, 1.65)
    ax.set_yticks([1.08, 0.42])
    ax.set_yticklabels(["Izq.", "Der."])
    ax.set_xlabel("Tiempo del segmento (s)", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Fases de marcha · línea temporal", fontsize=10)

    # Usar solo tres colores del ciclo por defecto de Matplotlib.
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2"])
    phase_color = {
        "Apoyo": cycle_colors[0 % len(cycle_colors)],
        "Propulsión": cycle_colors[1 % len(cycle_colors)],
        "Swing": cycle_colors[2 % len(cycle_colors)],
    }

    legend_seen = set()
    for cycles, y in [(left_cycles,1.08),(right_cycles,0.42)]:
        for cyc in cycles or []:
            # Agrupar subfases contiguas por macrofase.
            raw = _phase_intervals_for_cycle(cyc, fps, fmin)
            grouped = []
            for t0,t1,label in raw:
                macro = _macro_phase(label)
                if grouped and grouped[-1][2] == macro and abs(grouped[-1][1]-t0) < 1e-6:
                    grouped[-1] = (grouped[-1][0], t1, macro)
                else:
                    grouped.append((t0,t1,macro))

            for t0,t1,macro in grouped:
                a=max(0.0,t0); b=min(duration,t1)
                if b<=a:
                    continue
                label_for_legend = macro if macro not in legend_seen else None
                ax.barh(
                    y,b-a,left=a,height=0.46,
                    alpha=0.55, edgecolor="white", linewidth=0.6,
                    color=phase_color.get(macro),
                    label=label_for_legend
                )
                legend_seen.add(macro)

            # IC como marca fina de inicio de ciclo.
            tic = (float(cyc["ic_frame"])-float(fmin))/float(fps)
            if 0 <= tic <= duration:
                ax.vlines(tic, y-0.28, y+0.28, linewidth=0.55, alpha=0.45)

    tcur=(float(current_frame)-float(fmin))/float(fps)
    if 0<=tcur<=duration:
        ax.axvline(tcur,linewidth=2.0)
        ax.text(tcur,1.52,f"{tcur:.2f} s",ha="center",va="bottom",fontsize=7)

    if legend_seen:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5,-0.23),
                  ncol=3, frameon=False, fontsize=7)

    fig.tight_layout(pad=0.55)
    return fig


def _phase_context_text(cycles, frame):
    cyc = _find_cycle_for_frame(cycles, frame) if cycles else None
    if cyc is None:
        return "Entre ciclos", np.nan
    info = _cycle_phase_for_frame(cyc, frame)
    if info is None:
        return "Entre ciclos", np.nan
    return info[1], info[0]


def _all_event_frames(left_cycles, right_cycles):
    vals = []
    for side, cycles in [("I", left_cycles or []), ("D", right_cycles or [])]:
        for c in cycles:
            vals.append((int(c["ic_frame"]), f"{side} · IC"))
            if c.get("to_frame") is not None:
                vals.append((int(c["to_frame"]), f"{side} · TO"))
            vals.append((int(c["next_ic_frame"]), f"{side} · IC"))
    # Deduplicar por frame/label y ordenar.
    vals = sorted(set(vals), key=lambda x: (x[0], x[1]))
    return vals


def _nearest_cycle_for_frame(cycles, frame):
    if not cycles:
        return None
    inside = _find_cycle_for_frame(cycles, frame)
    if inside is not None:
        return inside
    return min(
        cycles,
        key=lambda c: abs(((c["ic_frame"]+c["next_ic_frame"])/2.0)-float(frame))
    )


def _phase_band_figure(cycle_left, cycle_right, current_frame=None):
    """
    v0.10.6 · Cada fila usa SU propio 0–100% y su propio cursor sincronizado
    al mismo frame de vídeo.
    """
    fig, ax = plt.subplots(figsize=(8.3, 1.55))
    ax.set_xlim(0,100)
    ax.set_ylim(-0.15,1.7)
    ax.set_yticks([1.12,0.38])
    ax.set_yticklabels(["Izq.","Der."])
    ax.set_xlabel("Ciclo (%)", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Fases sincronizadas", fontsize=10)

    for cyc,y,label in [(cycle_left,1.12,"I"),(cycle_right,0.38,"D")]:
        stance=float(cyc.get("stance_pct",60.0)) if cyc else 60.0
        for a,b,name in _phase_segments_for_stance(stance):
            if b<=a:
                continue
            ax.barh(y,b-a,left=a,height=0.48,alpha=0.30,edgecolor="black",linewidth=0.35)
            if (b-a)>=9:
                ax.text((a+b)/2,y,name,ha="center",va="center",fontsize=6.0)
        if cyc:
            ax.axvline(stance,linewidth=0.9,linestyle="--")
            if current_frame is not None and cyc["next_ic_frame"]>cyc["ic_frame"]:
                pct=(float(current_frame)-cyc["ic_frame"])/(cyc["next_ic_frame"]-cyc["ic_frame"])*100.0
                if 0<=pct<=100:
                    ax.axvline(pct,linewidth=1.6,linestyle="-" if label=="I" else ":")
                    ax.text(pct,y+0.31,f"{label} {pct:.0f}%",ha="center",va="bottom",fontsize=6.5)

    fig.tight_layout(pad=0.5)
    return fig


def _kinematic_cycle_figure(df_left, df_right, variable_left=None, variable_right=None,
                              left_pct=None, right_pct=None):
    fig, ax = plt.subplots(figsize=(8.3,2.8))
    plotted=False

    def add(df,var,label):
        nonlocal plotted
        if df is None or df.empty or var not in df.columns:
            return
        x=pd.to_numeric(df["cycle_pct"],errors="coerce").to_numpy(float)
        y=pd.to_numeric(df[var],errors="coerce").to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        if ok.sum()>=2:
            ax.plot(x[ok],y[ok],label=label,linewidth=1.8)
            plotted=True

    add(df_left,variable_left,f"Izq · {variable_left}")
    add(df_right,variable_right,f"Der · {variable_right}")

    if left_pct is not None and np.isfinite(left_pct) and 0<=left_pct<=100:
        ax.axvline(float(left_pct),linewidth=1.5,linestyle="-",label=f"Cursor I {left_pct:.0f}%")
    if right_pct is not None and np.isfinite(right_pct) and 0<=right_pct<=100:
        ax.axvline(float(right_pct),linewidth=1.5,linestyle=":",label=f"Cursor D {right_pct:.0f}%")

    ax.set_xlim(0,100)
    ax.set_xlabel("Ciclo (%)",fontsize=8)
    ax.set_ylabel("Magnitud 2D",fontsize=8)
    ax.tick_params(axis="both",labelsize=8)
    ax.set_title("Curva cinemática normalizada",fontsize=10)
    ax.grid(True,alpha=0.22)

    if plotted:
        ax.legend(loc="best",fontsize=6.5,ncol=2)
    else:
        ax.text(50,0.5,"Sin suficientes datos finitos para esta variable/ciclo",
                ha="center",va="center",transform=ax.get_xaxis_transform(),fontsize=8.5)
        ax.set_yticks([])
    fig.tight_layout(pad=0.6)
    return fig


def _video_frame_with_phase(video_path, frame_no, seg, left_cycle, right_cycle):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    # Overlay keypoints/segments from tracking if row is available.
    if seg is not None and not seg.empty:
        row_match = seg.iloc[(seg["frame"]-int(frame_no)).abs().argsort()[:1]]
        if not row_match.empty:
            row = row_match.iloc[0]
            pairs = [
                ("LShoulder","RShoulder"),("LShoulder","LHip"),("RShoulder","RHip"),
                ("LHip","RHip"),("LHip","LKnee"),("LKnee","LAnkle"),
                ("RHip","RKnee"),("RKnee","RAnkle"),
            ]
            pts = {}
            for name in HALPE26:
                xk,yk,sk=f"{name}_x",f"{name}_y",f"{name}_score"
                if xk in row and yk in row and sk in row and np.isfinite(row[xk]) and np.isfinite(row[yk]) and float(row[sk])>=0.15:
                    pts[name]=(int(round(row[xk])),int(round(row[yk])))
            for a,b in pairs:
                if a in pts and b in pts:
                    cv2.line(frame,pts[a],pts[b],(255,255,255),2,cv2.LINE_AA)
            for p in pts.values():
                cv2.circle(frame,p,4,(255,255,255),-1,cv2.LINE_AA)

    lines=[]
    for cyc,label in [(left_cycle,"IZQ"),(right_cycle,"DER")]:
        info=_cycle_phase_for_frame(cyc,int(frame_no)) if cyc else None
        if info:
            pct,phase=info
            lines.append(f"{label}: {phase} · {pct:.1f}%")
    if not lines:
        lines=["Fuera del ciclo seleccionado"]

    y=35
    for line in lines:
        cv2.putText(frame,line,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(frame,line,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2,cv2.LINE_AA)
        y+=32

    ok, enc = cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),92])
    return enc.tobytes() if ok else None


def _manual_event_editor(cycle, prefix):
    """
    Returns possibly corrected IC/TO/nextIC values using session_state.
    """
    if not cycle:
        return cycle
    ic_key=f"{prefix}_ic"
    to_key=f"{prefix}_to"
    nic_key=f"{prefix}_nic"
    if ic_key not in st.session_state:
        st.session_state[ic_key]=int(cycle["ic_frame"])
    if to_key not in st.session_state:
        st.session_state[to_key]=int(cycle["to_frame"]) if cycle.get("to_frame") is not None else int(round((cycle["ic_frame"]+cycle["next_ic_frame"])*0.6))
    if nic_key not in st.session_state:
        st.session_state[nic_key]=int(cycle["next_ic_frame"])

    a,b,c = st.columns(3)
    ic = a.number_input("IC", min_value=0, value=int(st.session_state[ic_key]), step=1, key=ic_key+"_widget")
    to = b.number_input("TO", min_value=0, value=int(st.session_state[to_key]), step=1, key=to_key+"_widget")
    nic = c.number_input("IC siguiente", min_value=0, value=int(st.session_state[nic_key]), step=1, key=nic_key+"_widget")
    if ic < to < nic:
        cycle = dict(cycle)
        cycle["ic_frame"]=int(ic)
        cycle["to_frame"]=int(to)
        cycle["next_ic_frame"]=int(nic)
        cycle["cycle_s"]=(nic-ic)  # se recalcula en segundos en la tabla usando fps_cycle
        cycle["stance_pct"]=(to-ic)/(nic-ic)*100.0
        st.session_state[ic_key]=int(ic)
        st.session_state[to_key]=int(to)
        st.session_state[nic_key]=int(nic)
    return cycle


APP_VERSION = "0.10.9-online"
TMP_ROOT = Path(tempfile.gettempdir()) / "physiosentinel_gait_online" / "sessions"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

HALPE26 = {
    "Nose": 0,
    "LShoulder": 5, "RShoulder": 6,
    "LElbow": 7, "RElbow": 8,
    "LWrist": 9, "RWrist": 10,
    "LHip": 11, "RHip": 12,
    "LKnee": 13, "RKnee": 14,
    "LAnkle": 15, "RAnkle": 16,
    "Head": 17, "Neck": 18, "Hip": 19,
    "LBigToe": 20, "RBigToe": 21,
    "LSmallToe": 22, "RSmallToe": 23,
    "LHeel": 24, "RHeel": 25,
}
LOWER_BODY = ["LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe"]
FOOT_POINTS = ["LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe"]
UPPER_BODY = ["LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"]
ASSISTIVE_OPTIONS = ["Sin ayuda", "Bastón", "1 muleta", "2 muletas", "Caminador", "Rollator", "Otra"]
SKELETON = [
    ("LShoulder", "RShoulder"), ("LShoulder", "LHip"), ("RShoulder", "RHip"), ("LHip", "RHip"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"),
    ("Neck", "Hip"), ("Nose", "Neck"),
]

st.set_page_config(page_title="PhysioSentinel Gait", page_icon="🚶", layout="wide")

# ------------------------- biblioteca de referencias -------------------------
# IMPORTANTE: estas bandas son referencias publicadas/contextuales, no umbrales diagnósticos.
# La comparabilidad depende de edad, sexo, velocidad, protocolo y sistema de medida.
REFERENCE_LIBRARY_VERSION = "2026.09.03"
REFERENCE_LIBRARY = {
    "cadence_exp": {
        "label": "Cadencia",
        "low": 114.95, "high": 118.35, "unit": "pasos/min",
        "population": "Adultos aparentemente sanos, marcha habitual al aire libre",
        "method": "Metaanálisis; media agrupada 116,65 pasos/min (IC95% 114,95–118,35)",
        "source": "Tudor-Locke et al., Sports Medicine 2021",
        "doi": "10.1007/s40279-020-01351-3",
        "applicability": "Contextual. El IC de la media no equivale a un intervalo individual de normalidad. No usar como umbral diagnóstico, especialmente en marcha neurológica."
    },
    "lateral_cadence_exp": {
        "label": "Cadencia lateral",
        "low": 114.95, "high": 118.35, "unit": "pasos/min",
        "population": "Adultos aparentemente sanos, marcha habitual al aire libre",
        "method": "Metaanálisis; media agrupada 116,65 pasos/min (IC95% 114,95–118,35)",
        "source": "Tudor-Locke et al., Sports Medicine 2021",
        "doi": "10.1007/s40279-020-01351-3",
        "applicability": "Contextual. El IC de la media no equivale a un intervalo individual de normalidad. No usar como umbral diagnóstico."
    },
}


# Referencias contextuales adicionales. Se usan solo cuando la definición de la métrica es comparable.
# Para proxies 2D sin equivalencia validada se muestra explícitamente que no existe un rango normativo transferible.
REFERENCE_CONTEXT = {
    "regularity_cv": "Referencia análoga: CV de tiempo de paso ~3.35% como umbral de variabilidad patológica en metaanálisis; no es idéntico a la alternancia 2D de esta app (König et al., 2019, PMID 31639377).",
    "temporal_asymmetry_exp": "Sin umbral universal transferible a este detector 2D. En laboratorio, la simetría del tiempo de apoyo suele ser cercana a 1:1; una cohorte joven femenina reportó ~0.7% a velocidad preferida (PMCID PMC6335661).",
    "stance_asymmetry_2d": "Referencia contextual: el tiempo de apoyo sano es aproximadamente simétrico entre lados; los límites dependen del índice y del sistema de medida. Este valor es una estimación 2D experimental.",
    "stance_pct_2d": "Referencia temporal clásica en marcha adulta confortable: apoyo ~60% del ciclo; contextual, no umbral diagnóstico y dependiente de velocidad/patología.",
    "swing_pct_2d": "Referencia temporal clásica en marcha adulta confortable: oscilación ~40% del ciclo; contextual, no umbral diagnóstico.",
    "double_support_pct_2d": "Referencia temporal clásica en marcha adulta confortable: doble apoyo total ~20–24% del ciclo (dos periodos de ~10–12%); aumenta habitualmente al disminuir la velocidad.",
    "double_support_time_2d": "Sin duración absoluta universal; depende de la duración del ciclo. Referencia porcentual contextual: ~20–24% del ciclo en marcha adulta confortable.",
    "swing_time_l_2d": "Sin duración absoluta universal; referencia proporcional contextual: oscilación ~40% del ciclo en marcha adulta confortable.",
    "swing_time_r_2d": "Sin duración absoluta universal; referencia proporcional contextual: oscilación ~40% del ciclo en marcha adulta confortable.",
    "swing_asymmetry_2d": "Sin umbral 2D universal validado para este detector. En marcha simétrica los tiempos de oscilación deberían aproximarse entre extremidades; interpretar longitudinalmente.",
    "initial_contact_foot_l_deg": "Sin rango normativo 2D universal transferible. La orientación en contacto inicial depende del plano de cámara, velocidad, calzado y estrategia de contacto.",
    "initial_contact_foot_r_deg": "Sin rango normativo 2D universal transferible. La orientación en contacto inicial depende del plano de cámara, velocidad, calzado y estrategia de contacto.",
    "initial_contact_rearfoot_l_deg": "Sin umbral 2D validado universal para inclinación del retropié en contacto inicial; descriptor proyectado.",
    "initial_contact_rearfoot_r_deg": "Sin umbral 2D validado universal para inclinación del retropié en contacto inicial; descriptor proyectado.",
    "loading_knee_l_deg": "Sin umbral 2D universal. Descriptor de estabilidad/alineación de rodilla durante la ventana de respuesta a la carga.",
    "loading_knee_r_deg": "Sin umbral 2D universal. Descriptor de estabilidad/alineación de rodilla durante la ventana de respuesta a la carga.",
    "terminal_foot_l_deg": "Sin rango normativo 2D universal transferible para orientación distal en pre-oscilación/despegue.",
    "terminal_foot_r_deg": "Sin rango normativo 2D universal transferible para orientación distal en pre-oscilación/despegue.",
    "pelvis_obliquity_rom": "No existe un rango normativo directamente transferible al ROM dinámico HALPE26 2D. La oblicuidad pélvica estática en población sana se ha descrito entre 0–5.6°, pero no equivale a este ROM dinámico (PMID 37254005).",
    "shoulder_obliquity_rom": "Sin rango normativo validado para este proxy 2D markerless; priorizar comparación intraindividual.",
    "trunk_lateral_lean_rom": "Sin rango normativo validado para este proxy 2D markerless; depende de velocidad, tarea y estrategia compensatoria.",
    "shoulder_pelvis_rel_rom": "Sin rango normativo validado para este acoplamiento 2D; usar como descriptor longitudinal.",
    "com_lateral_excursion_cm": "Sin rango universal: la excursión lateral del CoM depende de velocidad, ancho de paso y método. Solo se expresa en cm si existe escala espacial calibrada.",
    "bos_width_cm": "Sin rango universal transferible: el ancho de base depende de antropometría, edad y velocidad. Interpretar con contexto y basal individual.",
    "trendelenburg_drop_l_deg": "Sin umbral diagnóstico 2D universal. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente si es relevante.",
    "trendelenburg_drop_r_deg": "Sin umbral diagnóstico 2D universal. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente si es relevante.",
    "dynamic_knee_valgus_l_deg": "Sin rango normativo validado para HALPE26 2D. El valgo dinámico es multiplanar; este valor es solo desviación medial proyectada.",
    "dynamic_knee_valgus_r_deg": "Sin rango normativo validado para HALPE26 2D. El valgo dinámico es multiplanar; este valor es solo desviación medial proyectada.",
    "trunk_pelvis_coupling_r": "Sin banda normativa universal para este coeficiente 2D. Valores cercanos a +1 indican acoplamiento en fase y cercanos a -1, contrafase; el significado clínico depende de la tarea.",
    "trunk_pelvis_phase_deg": "Sin banda normativa universal para este desfase 2D. Interpretar longitudinalmente y junto con la calidad del tracking.",
}

def reference_text_for_metric(key):
    base = key
    for pref in ("front_", "lateral_"):
        if base.startswith(pref):
            base = base[len(pref):]
    ref = REFERENCE_LIBRARY.get(key) or REFERENCE_LIBRARY.get(base)
    if ref:
        return f"Referencia poblacional: {ref['low']:.2f}–{ref['high']:.2f} {ref['unit']} ({ref['population']}); contextual, no umbral diagnóstico."
    if base in REFERENCE_CONTEXT:
        return REFERENCE_CONTEXT[base]
    if any(tok in base for tok in ["knee_flex","hip_flex","ankle_angle","shoulder_elev","frontal_knee_dev","foot_progress","rearfoot_tilt","base_width_relative"]):
        return "Sin rango normativo validado directamente transferible a esta métrica 2D proyectada; usar comparación intraindividual y contexto clínico."
    if base in {"tracking_mean","good_frames_pct","foot_visibility_pct","upper_visibility_pct","step_count_consistency_error_pct"}:
        return "Referencia metodológica interna de calidad; no es una variable clínica normativa."
    if base in {"step_events_detected","segment_duration_s","expected_steps_from_cadence","cadence_count_segment","alternation_interval"}:
        return "Sin rango normativo clínico único; variable dependiente de la tarea/segmento y usada para consistencia interna."
    return "Sin rango normativo estandarizado validado para esta definición y método; interpretar respecto al basal individual y al contexto."

REFERENCE_SOURCES = [
    {"Fuente":"Kreusch et al. 2026","Uso":"Mapa de evidencia normativa en adultos sanos 18–65 años (105 estudios; 11.764 participantes)","DOI":"10.1016/j.gaitpost.2026.110176"},
    {"Fuente":"Herssens et al. 2018","Uso":"Parámetros espaciotemporales y variabilidad a lo largo de la vida adulta","DOI":"10.1016/j.gaitpost.2018.06.012"},
    {"Fuente":"Tudor-Locke et al. 2021","Uso":"Velocidad/cadencia de marcha habitual y otros ritmos en adultos sanos","DOI":"10.1007/s40279-020-01351-3"},
    {"Fuente":"Fukuchi et al. 2019","Uso":"Efecto de la velocidad sobre parámetros espaciotemporales, cinemática y cinética","DOI":"10.1186/s12984-019-0559-8"},
    {"Fuente":"Sato et al. 2023","Uso":"Referencias preliminares markerless de tronco y miembro inferior en mayores sanos japoneses","DOI":"10.1298/ptr.E10247"},
]

def reference_for_metric(key):
    return REFERENCE_LIBRARY.get(key)

def reference_position(value, ref):
    if ref is None or value is None or not np.isfinite(float(value)):
        return "Sin referencia"
    v=float(value)
    if v < ref["low"]: return "Por debajo de la banda publicada"
    if v > ref["high"]: return "Por encima de la banda publicada"
    return "Dentro de la banda publicada"

# ------------------------- seguridad / secretos -------------------------
def secret(name, default=None):
    # Streamlit Community Cloud: st.secrets
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass

    # Hugging Face Spaces / Docker: secrets como variables de entorno
    return os.getenv(name, default)

SUPABASE_URL = (secret("SUPABASE_URL", "") or "").rstrip("/")
SUPABASE_KEY = secret("SUPABASE_SERVICE_ROLE_KEY", "") or ""
APP_PASSWORD = secret("GAIT_APP_PASSWORD", "") or ""


def require_password():
    if not APP_PASSWORD:
        st.warning("⚠️ GAIT_APP_PASSWORD no está configurada. Modo de prueba sin control de acceso.")
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("PhysioSentinel Gait")
    st.caption("Acceso protegido")
    pwd = st.text_input("Contraseña", type="password")
    if st.button("Entrar", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False

if not require_password():
    st.stop()

# ------------------------- Supabase REST -------------------------
def sb_ready():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def sb_headers(extra_prefer=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra_prefer:
        h["Prefer"] = extra_prefer
    return h


def sb_request(method, table, params=None, payload=None, prefer=None, timeout=30):
    if not sb_ready():
        raise RuntimeError("Supabase no está configurado en Streamlit Secrets.")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.request(method, url, headers=sb_headers(prefer), params=params, json=payload, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text[:800]}")
    if not r.text.strip():
        return None
    try:
        return r.json()
    except Exception:
        return r.text


def sb_upsert_patient(code):
    data = sb_request(
        "POST", "gait_patients",
        params={"on_conflict": "code"},
        payload={"code": code},
        prefer="resolution=merge-duplicates,return=representation",
    )
    if not data:
        data = sb_request("GET", "gait_patients", params={"code": f"eq.{code}", "select": "id,code"})
    return data[0]["id"]


def sb_create_session(code, record_name, mode, view, meta, assistive_device="Sin ayuda", frontal_orientation="No especificada", meta2=None, calibration_profile_name=None):
    patient_id = sb_upsert_patient(code)
    session_id = str(uuid.uuid4())
    payload = {
        "id": session_id,
        "patient_id": patient_id,
        "record_name": record_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "view": view or "",
        "assistive_device": assistive_device or "Sin ayuda",
        "assisted_gait": bool((assistive_device or "Sin ayuda") != "Sin ayuda"),
        "frontal_orientation": frontal_orientation or "No especificada",
        "fps": float(meta.get("fps", 0)) if meta else None,
        "frames": int(meta.get("frames", 0)) if meta else None,
        "duration_s": float(meta.get("duration", 0)) if meta else None,
        "fps_cam2": float(meta2.get("fps", 0)) if meta2 else None,
        "frames_cam2": int(meta2.get("frames", 0)) if meta2 else None,
        "duration_cam2_s": float(meta2.get("duration", 0)) if meta2 else None,
        "calibration_profile_name": calibration_profile_name or None,
        "video_persisted": False,
        "app_version": APP_VERSION,
    }
    sb_request("POST", "gait_sessions", payload=payload, prefer="return=minimal")
    return session_id


def sb_save_metrics(session_id, metrics, start_s, end_s):
    payload = []
    for m in metrics:
        v = m.get("value")
        if v is not None:
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = None
            except Exception:
                v = None
        payload.append({
            "session_id": session_id,
            "metric_key": m["key"],
            "metric_label": m["label"],
            "value": v,
            "unit": m.get("unit", ""),
            "quality": m.get("quality", ""),
            "notes": m.get("notes", ""),
        })
    sb_request(
        "POST", "gait_metrics",
        params={"on_conflict": "session_id,metric_key"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
        timeout=60,
    )
    sb_request(
        "PATCH", "gait_sessions",
        params={"id": f"eq.{session_id}"},
        payload={"segment_start_s": float(start_s), "segment_end_s": float(end_s), "analysis_status": "completed"},
        prefer="return=minimal",
    )


def sb_list_patients():
    if not sb_ready():
        return pd.DataFrame()
    data = sb_request("GET", "gait_patients", params={"select": "id,code,created_at", "order": "code.asc"}) or []
    return pd.DataFrame(data)


def sb_list_calibrations():
    if not sb_ready():
        return []
    try:
        return sb_request(
            "GET", "gait_calibrations",
            params={"select": "id,name,camera_count,notes,created_at", "order": "name.asc"},
        ) or []
    except Exception:
        return []


def sb_get_calibration(name):
    if not sb_ready() or not name:
        return None
    data = sb_request(
        "GET", "gait_calibrations",
        params={"name": f"eq.{name}", "select": "id,name,camera_count,content_toml,notes,created_at", "limit": 1},
    ) or []
    return data[0] if data else None


def sb_upsert_calibration(name, content_toml, notes="", camera_count=2):
    payload = {
        "name": name.strip(),
        "camera_count": int(camera_count),
        "content_toml": content_toml,
        "notes": notes or "",
    }
    data = sb_request(
        "POST", "gait_calibrations",
        params={"on_conflict": "name"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    return data[0] if data else None


def sb_update_session_3d(session_id, **fields):
    if not session_id or not fields:
        return
    clean = {k: v for k, v in fields.items() if v is not None}
    if clean:
        sb_request(
            "PATCH", "gait_sessions",
            params={"id": f"eq.{session_id}"},
            payload=clean,
            prefer="return=minimal",
        )


def sb_update_session_identity(session_id, patient_code=None, record_name=None):
    """
    v0.9.3 · Actualiza únicamente campos ya existentes en Supabase.
    Edad/sexo/fecha clínica permanecen en la sesión local y en el informe
    hasta que el esquema remoto disponga de columnas específicas.
    """
    if not session_id:
        return
    payload = {}
    if patient_code:
        patient_id = sb_upsert_patient(str(patient_code).strip())
        payload["patient_id"] = patient_id
    if record_name is not None:
        payload["record_name"] = str(record_name).strip()
    if payload:
        sb_request(
            "PATCH", "gait_sessions",
            params={"id": f"eq.{session_id}"},
            payload=payload,
            prefer="return=minimal",
        )


def sb_delete_session(session_id):
    """Elimina de forma permanente un registro y sus métricas asociadas.

    Se borran primero gait_metrics y después gait_sessions para funcionar
    incluso si la FK no tiene ON DELETE CASCADE. No elimina al paciente.
    """
    if not session_id:
        raise ValueError("Falta el identificador de la sesión.")
    sb_request(
        "DELETE", "gait_metrics",
        params={"session_id": f"eq.{session_id}"},
        prefer="return=minimal",
    )
    sb_request(
        "DELETE", "gait_sessions",
        params={"id": f"eq.{session_id}"},
        prefer="return=minimal",
    )
    return True


def sb_patient_history(code):
    if not sb_ready():
        return pd.DataFrame()
    p = sb_request("GET", "gait_patients", params={"code": f"eq.{code}", "select": "id,code", "limit": 1}) or []
    if not p:
        return pd.DataFrame()
    pid = p[0]["id"]
    sessions = sb_request(
        "GET", "gait_sessions",
        params={"patient_id": f"eq.{pid}", "select": "id,created_at,record_name,mode,view,assistive_device,assisted_gait,frontal_orientation,fps,frames,duration_s,fps_cam2,frames_cam2,duration_cam2_s,sync_offset_s,sync_correlation,sync_quality,calibration_profile_name,ready_3d,segment_start_s,segment_end_s,analysis_status", "order": "created_at.asc"},
    ) or []
    if not sessions:
        return pd.DataFrame()
    ids = ",".join(s["id"] for s in sessions)
    metrics = sb_request(
        "GET", "gait_metrics",
        params={"session_id": f"in.({ids})", "select": "session_id,metric_key,metric_label,value,unit,quality,notes"},
    ) or []
    sdf = pd.DataFrame(sessions)
    mdf = pd.DataFrame(metrics)
    if mdf.empty:
        return pd.DataFrame()
    return sdf.merge(mdf, left_on="id", right_on="session_id", how="inner")

# ------------------------- utilidades vídeo -------------------------
def safe_name(text):
    text = (text or "sesion").strip()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)[:60] or "sesion"


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps, "frames": frames, "width": width, "height": height,
        "duration": frames / fps if fps > 0 else 0,
        "orientation": "Vertical" if height > width else "Horizontal",
    }


def save_upload(uploaded, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())


def create_temp_session(patient, record):
    old = st.session_state.get("session_dir")
    if old:
        try:
            shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass
    folder = TMP_ROOT / f"{safe_name(patient)}_{safe_name(record)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    return folder


def prepare_config(session_dir):
    cfg = {
        "project": {
            "project_dir": str(session_dir),
            "multi_person": False,
            "participant_height": "auto",
            "participant_mass": 70,
            "frame_rate": "auto",
            "frame_range": "auto",
        },
        "pose": {
            "pose_model": "Body_with_feet",
            "mode": "balanced",
            "det_frequency": 4,
            "device": "auto",
            "backend": "auto",
            "display_detection": False,
            "overwrite_pose": True,
            "save_video": "to_video",
            "output_format": "openpose",
            "tracking_mode": "sports2d",
        },
    }
    path = session_dir / "Config.toml"
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    return path


def run_pose2sim(config_path):
    from Pose2Sim import Pose2Sim
    Pose2Sim.poseEstimation(str(config_path))


def find_pose_json_dir(session_dir, cam="cam01"):
    pose_dir = session_dir / "pose"
    if not pose_dir.exists():
        return None
    preferred = sorted([p for p in pose_dir.rglob(f"{cam}*_json") if p.is_dir()])
    if preferred:
        return preferred[0]
    candidates = sorted([p for p in pose_dir.rglob("*_json") if p.is_dir()])
    return candidates[0] if candidates else None


def parse_frame_number(path, fallback):
    m = re.search(r"(\d+)(?=\.json$)", path.name)
    return int(m.group(1)) if m else fallback


def load_pose_dataframe(json_dir):
    rows = []
    for i, path in enumerate(sorted(json_dir.glob("*.json"))):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if not people:
                continue
            pts = people[0].get("pose_keypoints_2d", [])
            if len(pts) < 26 * 3:
                continue
            row = {"frame": parse_frame_number(path, i)}
            for name, idx in HALPE26.items():
                base = idx * 3
                row[f"{name}_x"] = float(pts[base])
                row[f"{name}_y"] = float(pts[base + 1])
                row[f"{name}_score"] = float(pts[base + 2])
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True) if rows else pd.DataFrame()


# ------------------------- v0.9.0 · selección multipersona -------------------------
def _person_points(person):
    pts = person.get("pose_keypoints_2d", []) if isinstance(person, dict) else []
    if len(pts) < 26 * 3:
        return None
    return np.asarray(pts[:26*3], dtype=float).reshape(26, 3)


def _descriptor_from_points(pts, score_thr=0.20):
    if pts is None or pts.shape[0] < 26:
        return None
    valid = np.isfinite(pts[:,0]) & np.isfinite(pts[:,1]) & np.isfinite(pts[:,2]) & (pts[:,2] >= score_thr)
    if int(valid.sum()) < 7:
        return None

    xy = pts[:, :2]
    # Centro robusto: pelvis/hips/shoulders si son visibles; si no, mediana global.
    core_idx = [HALPE26["Hip"], HALPE26["LHip"], HALPE26["RHip"], HALPE26["LShoulder"], HALPE26["RShoulder"]]
    core_valid = [i for i in core_idx if valid[i]]
    if core_valid:
        center = np.nanmedian(xy[core_valid], axis=0)
    else:
        center = np.nanmedian(xy[valid], axis=0)

    xv, yv = xy[valid,0], xy[valid,1]
    x1, x2 = float(np.nanmin(xv)), float(np.nanmax(xv))
    y1, y2 = float(np.nanmin(yv)), float(np.nanmax(yv))
    w, h = max(1.0, x2-x1), max(1.0, y2-y1)
    scale = float(max(h, w, np.hypot(w, h) * 0.70))
    shape = np.full((26,2), np.nan, dtype=float)
    shape[valid] = (xy[valid] - center) / max(scale, 1.0)

    return {
        "center": np.asarray(center, dtype=float),
        "bbox": np.asarray([x1,y1,x2,y2], dtype=float),
        "scale": scale,
        "shape": shape,
        "valid": valid,
        "mean_score": float(np.nanmean(pts[valid,2])),
        "n_valid": int(valid.sum()),
    }


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2])); y2=min(float(a[3]),float(b[3]))
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1)
    inter=iw*ih
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    den=aa+bb-inter
    return float(inter/den) if den>0 else 0.0


def _shape_distance(a, b):
    va = a.get("shape"); vb = b.get("shape")
    if va is None or vb is None:
        return 1.0
    ok = np.isfinite(va[:,0]) & np.isfinite(va[:,1]) & np.isfinite(vb[:,0]) & np.isfinite(vb[:,1])
    if int(ok.sum()) < 5:
        return 1.0
    d = np.linalg.norm(va[ok]-vb[ok], axis=1)
    return float(np.nanmedian(d))


def _identity_cost(prev, cand, predicted_center=None):
    if prev is None or cand is None:
        return np.inf
    pc = np.asarray(predicted_center if predicted_center is not None else prev["center"], dtype=float)
    scale = max(float(prev.get("scale",1.0)), float(cand.get("scale",1.0)), 1.0)
    center_d = float(np.linalg.norm(np.asarray(cand["center"])-pc) / scale)
    iou_pen = 1.0 - _bbox_iou(prev.get("bbox"), cand.get("bbox"))
    scale_pen = abs(float(np.log(max(cand.get("scale",1.0),1.0) / max(prev.get("scale",1.0),1.0))))
    shape_pen = min(2.0, _shape_distance(prev, cand))
    conf_pen = max(0.0, 0.55 - float(cand.get("mean_score",0.0)))
    return float(0.42*center_d + 0.23*iou_pen + 0.14*scale_pen + 0.17*shape_pen + 0.04*conf_pen)


def _frame_people(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    out=[]
    for idx, person in enumerate(data.get("people", []) or []):
        pts=_person_points(person)
        desc=_descriptor_from_points(pts)
        if desc is None:
            continue
        out.append({"person_index":int(idx), "pts":pts, "desc":desc})
    return out


def scan_subject_candidates(json_dir):
    """
    Escoge automáticamente UN FOTOGRAMA de selección con la mayor cantidad de
    sujetos suficientemente visibles. No elige al paciente: solo prepara las
    opciones para que el clínico lo seleccione manualmente.
    """
    paths=sorted(Path(json_dir).glob("*.json"))
    if not paths:
        return None

    best=None
    # Preferir la primera mitad del ensayo para que la elección se haga antes
    # de cruces/giro, pero permitir cualquier frame si allí hay mejor visibilidad.
    scored=[]
    for i,p in enumerate(paths):
        people=_frame_people(p)
        if not people:
            continue
        n=len(people)
        quality=float(np.mean([x["desc"]["mean_score"] for x in people]))
        central_bonus=0.05 if i <= max(1, int(0.55*len(paths))) else 0.0
        score=n*10.0 + quality + central_bonus
        scored.append((score, -i, p, people))
    if not scored:
        return None
    scored.sort(key=lambda x:(x[0],x[1]), reverse=True)
    _,_,p,people=scored[0]
    frame_no=parse_frame_number(p,0)

    # Etiquetas estables para la UI: de izquierda a derecha en el frame elegido.
    ordered=sorted(people, key=lambda x: float(x["desc"]["center"][0]))
    candidates=[]
    for pos, item in enumerate(ordered, start=1):
        d=item["desc"]
        candidates.append({
            "label":f"Sujeto {pos}",
            "person_index":int(item["person_index"]),
            "center_x":float(d["center"][0]),
            "center_y":float(d["center"][1]),
            "bbox":[float(v) for v in d["bbox"]],
            "mean_score":float(d["mean_score"]),
            "n_valid":int(d["n_valid"]),
        })
    return {"frame":int(frame_no), "candidates":candidates, "max_people":int(len(candidates))}


def render_subject_preview(video_path, selection):
    if not selection:
        return None
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_no=int(selection["frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame=cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    for cand in selection["candidates"]:
        x1,y1,x2,y2=[int(round(v)) for v in cand["bbox"]]
        x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1]-1,x2); y2=min(frame.shape[0]-1,y2)
        # Sin colores clínicamente semánticos: etiquetas y cajas de alto contraste.
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3)
        cv2.putText(frame,cand["label"],(x1,max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(frame,cand["label"],(x1,max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
    ok, enc=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return enc.tobytes() if ok else None


def _row_from_tracked(frame_no, pts):
    row={"frame":int(frame_no)}
    for name,idx in HALPE26.items():
        row[f"{name}_x"]=float(pts[idx,0])
        row[f"{name}_y"]=float(pts[idx,1])
        row[f"{name}_score"]=float(pts[idx,2])
    return row



def _identity_anchor_guard(anchor_desc, prev_desc, cand_desc, gap=0):
    """
    v0.10.3 · Barrera adicional contra saltos paciente→acompañante.

    Devuelve (ok, reason). No identifica a la persona por apariencia; solo impide
    cambios geométricos bruscos incompatibles con una continuidad física razonable.
    """
    if anchor_desc is None or prev_desc is None or cand_desc is None:
        return False, "descriptor ausente"

    prev_scale = max(float(prev_desc.get("scale", 1.0)), 1.0)
    cand_scale = max(float(cand_desc.get("scale", 1.0)), 1.0)
    anchor_scale = max(float(anchor_desc.get("scale", 1.0)), 1.0)

    # Cambio brusco frame-a-frame: especialmente útil cuando el terapeuta
    # pasa muy cerca de cámara y aparece súbitamente mucho mayor.
    ratio_prev = cand_scale / prev_scale
    lo = 0.58 if gap <= 2 else 0.48
    hi = 1.72 if gap <= 2 else 1.95
    if not (lo <= ratio_prev <= hi):
        return False, "salto brusco de escala"

    center_jump = float(
        np.linalg.norm(np.asarray(cand_desc["center"]) - np.asarray(prev_desc["center"]))
        / max(prev_scale, cand_scale, 1.0)
    )
    max_jump = 0.78 if gap == 0 else min(1.15, 0.78 + 0.05 * min(gap, 7))
    if center_jump > max_jump:
        return False, "salto espacial excesivo"

    # La forma normalizada no necesita ser idéntica, pero un cambio extremo
    # frente al ancla seleccionada se trata como posible intercambio.
    anchor_shape = _shape_distance(anchor_desc, cand_desc)
    if np.isfinite(anchor_shape) and anchor_shape > 0.95:
        return False, "geometría incompatible con sujeto ancla"

    # Cambio de escala respecto al ancla puede ser grande por perspectiva,
    # pero no arbitrario. Este límite es deliberadamente amplio.
    ratio_anchor = cand_scale / anchor_scale
    if ratio_anchor < 0.28 or ratio_anchor > 3.8:
        return False, "escala incompatible con sujeto ancla"

    return True, "ok"


def _crop_frame_to_subject(frame, row, pad=0.38):
    """
    Recorta visualmente alrededor del sujeto TRACKING seleccionado.
    Solo afecta a la presentación de pestaña 9; no cambia las métricas.
    """
    if frame is None or row is None:
        return frame
    xs, ys = [], []
    for name in HALPE26:
        xk, yk, sk = f"{name}_x", f"{name}_y", f"{name}_score"
        try:
            if xk in row and yk in row and sk in row and float(row[sk]) >= 0.15:
                x, y = float(row[xk]), float(row[yk])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x); ys.append(y)
        except Exception:
            continue
    if len(xs) < 5:
        return frame

    h, w = frame.shape[:2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = max(40.0, x2-x1), max(80.0, y2-y1)
    px, py = bw*pad, bh*pad
    xa = max(0, int(round(x1-px)))
    xb = min(w, int(round(x2+px)))
    ya = max(0, int(round(y1-py)))
    yb = min(h, int(round(y2+py)))
    if xb-xa < 80 or yb-ya < 120:
        return frame
    return frame[ya:yb, xa:xb].copy()


def _draw_selected_tracking(frame, frame_no, seg, left_cycle=None, right_cycle=None, crop_subject=True):
    """
    Dibuja SOLO el tracking presente en `seg`, que en modo multipersona es el
    DataFrame del sujeto manualmente bloqueado. Nunca usa personas del vídeo anotado.
    """
    if frame is None:
        return None

    row = None
    if _valid_dataframe(seg):
        match = seg.iloc[(seg["frame"]-int(frame_no)).abs().argsort()[:1]]
        if not match.empty:
            row = match.iloc[0]

    # Dibujar sobre frame limpio.
    if row is not None:
        pairs = [
            ("LShoulder","RShoulder"),("LShoulder","LHip"),("RShoulder","RHip"),
            ("LHip","RHip"),("LHip","LKnee"),("LKnee","LAnkle"),
            ("RHip","RKnee"),("RKnee","RAnkle"),
            ("LAnkle","LHeel"),("LAnkle","LBigToe"),
            ("RAnkle","RHeel"),("RAnkle","RBigToe"),
        ]
        pts = {}
        for name in HALPE26:
            xk,yk,sk=f"{name}_x",f"{name}_y",f"{name}_score"
            try:
                if xk in row and yk in row and sk in row and float(row[sk]) >= 0.15:
                    x,y=float(row[xk]),float(row[yk])
                    if np.isfinite(x) and np.isfinite(y):
                        pts[name]=(int(round(x)),int(round(y)))
            except Exception:
                continue
        for a,b in pairs:
            if a in pts and b in pts:
                cv2.line(frame,pts[a],pts[b],(40,220,80),3,cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(frame,p,4,(0,190,255),-1,cv2.LINE_AA)

    # Fases antes del crop para conservar texto si no se recorta.
    phase_lines=[]
    for cyc,label in [(left_cycle,"IZQ"),(right_cycle,"DER")]:
        info=_cycle_phase_for_frame(cyc,int(frame_no)) if cyc else None
        if info:
            pct,phase=info
            phase_lines.append(f"{label}: {phase} · {pct:.1f}%")

    # Recortar después de dibujar tracking; el texto se añade al recorte.
    if crop_subject and row is not None:
        frame = _crop_frame_to_subject(frame, row, pad=0.42)

    y=30
    for line in phase_lines:
        cv2.putText(frame,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(frame,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)
        y += 27

    ok, enc=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return enc.tobytes() if ok else None


def load_pose_dataframe_tracked(json_dir, anchor_frame, selected_person_index):
    """
    Seguimiento de identidad bloqueado desde el sujeto elegido manualmente.

    - Sigue hacia delante y hacia atrás desde el frame de selección.
    - Combina continuidad espacial, tamaño corporal, IoU y forma esquelética.
    - Si dos candidatos son demasiado parecidos o el coste es excesivo,
      NO cambia silenciosamente de persona: excluye ese frame.
    """
    paths=sorted(Path(json_dir).glob("*.json"))
    if not paths:
        return pd.DataFrame(), {}

    frames=[]
    for i,p in enumerate(paths):
        fn=parse_frame_number(p,i)
        frames.append((int(fn),p))
    frame_to_pos={fn:i for i,(fn,_) in enumerate(frames)}
    if int(anchor_frame) not in frame_to_pos:
        # frame más cercano, por robustez ante nombres de archivo no contiguos
        anchor_frame=min(frame_to_pos, key=lambda x:abs(x-int(anchor_frame)))
    a_pos=frame_to_pos[int(anchor_frame)]

    anchor_people=_frame_people(frames[a_pos][1])
    anchor=next((x for x in anchor_people if int(x["person_index"])==int(selected_person_index)),None)
    if anchor is None:
        return pd.DataFrame(), {"quality":"No fiable","reason":"No se encontró el sujeto seleccionado en el frame ancla."}

    accepted={int(anchor_frame):anchor}
    ambiguous=set()
    missing=set()
    switches_prevented=0
    max_people=0

    def walk(indices):
        nonlocal switches_prevented,max_people
        prev=anchor
        prev2=None
        gap=0
        for pos in indices:
            fn,p=frames[pos]
            people=_frame_people(p)
            max_people=max(max_people,len(people))
            if not people:
                missing.add(fn); gap+=1
                continue

            pred=None
            if prev2 is not None and gap==0:
                v=np.asarray(prev["desc"]["center"])-np.asarray(prev2["desc"]["center"])
                pred=np.asarray(prev["desc"]["center"])+v

            scored=[]
            for cand in people:
                cost=_identity_cost(prev["desc"],cand["desc"],predicted_center=pred)
                scored.append((cost,cand))
            scored.sort(key=lambda x:x[0])
            best_cost,best=scored[0]
            second_cost=scored[1][0] if len(scored)>1 else np.inf
            margin=second_cost-best_cost

            # Umbral algo más permisivo tras una oclusión breve, pero nunca
            # permite un salto grande de identidad.
            max_cost=0.92 if gap==0 else min(1.15,0.92+0.025*min(gap,9))
            guard_ok, guard_reason = _identity_anchor_guard(
                anchor["desc"], prev["desc"], best["desc"], gap=gap
            )
            ambiguous_match = (
                best_cost > max_cost or
                (len(scored)>1 and margin < 0.14 and best_cost > 0.22) or
                (not guard_ok)
            )
            if ambiguous_match:
                ambiguous.add(fn)
                switches_prevented += 1 if len(scored)>1 else 0
                gap += 1
                # v0.10.3: nunca adoptar un candidato que requiera un salto
                # brusco de escala/posición/geometría respecto al paciente bloqueado.
                continue

            accepted[fn]=best
            prev2=prev
            prev=best
            gap=0

    walk(range(a_pos+1,len(frames)))
    # Reiniciar referencia para seguimiento hacia atrás.
    prev_anchor=anchor
    # La función walk mantiene variables locales prev, por lo que basta otra llamada
    walk(range(a_pos-1,-1,-1))

    rows=[]
    for fn,_ in frames:
        item=accepted.get(fn)
        if item is not None:
            rows.append(_row_from_tracked(fn,item["pts"]))

    df=pd.DataFrame(rows).sort_values("frame").reset_index(drop=True) if rows else pd.DataFrame()
    total=len(frames)
    reliable=len(rows)
    excluded=max(0,total-reliable)
    continuity=100.0*reliable/total if total else np.nan
    ambiguous_pct=100.0*len(ambiguous)/total if total else np.nan
    missing_pct=100.0*len(missing)/total if total else np.nan

    if np.isfinite(continuity):
        quality="Alta" if continuity>=90 else ("Moderada" if continuity>=75 else "Baja")
    else:
        quality="No fiable"
    info={
        "manual":True,
        "anchor_frame":int(anchor_frame),
        "selected_person_index":int(selected_person_index),
        "identity_continuity_pct":float(continuity) if np.isfinite(continuity) else np.nan,
        "ambiguous_excluded_pct":float(ambiguous_pct) if np.isfinite(ambiguous_pct) else np.nan,
        "missing_pose_pct":float(missing_pct) if np.isfinite(missing_pct) else np.nan,
        "frames_total":int(total),
        "frames_reliable":int(reliable),
        "frames_excluded":int(excluded),
        "ambiguous_frames":int(len(ambiguous)),
        "switches_prevented":int(switches_prevented),
        "max_people_detected":int(max_people),
        "quality":quality,
    }
    return df,info


def tracking_metrics(info, prefix="", view_label=""):
    if not info or not info.get("manual"):
        return []
    p=(prefix+"_" if prefix else "")
    view_note=f" ({view_label})" if view_label else ""
    return [
        {"key":p+"subject_manual_selection_flag","label":"Selección manual de sujeto"+view_note,"value":1.0,"unit":"bool","quality":"Bloqueado","notes":"v0.9.1: el paciente fue seleccionado explícitamente por el clínico antes del análisis biomecánico."},
        {"key":p+"identity_continuity_pct","label":"Continuidad de identidad"+view_note,"value":info.get("identity_continuity_pct"),"unit":"%","quality":info.get("quality","No fiable"),"notes":"Porcentaje de frames en los que la identidad seleccionada se mantuvo con confianza suficiente."},
        {"key":p+"identity_ambiguous_excluded_pct","label":"Frames ambiguos excluidos"+view_note,"value":info.get("ambiguous_excluded_pct"),"unit":"%","quality":"Control interno","notes":"Frames descartados por riesgo de confundir al paciente con otra persona."},
        {"key":p+"identity_frames_excluded","label":"Frames excluidos por identidad"+view_note,"value":info.get("frames_excluded"),"unit":"frames","quality":"Control interno","notes":"Incluye oclusiones, pérdidas de pose y emparejamientos ambiguos; nunca se sustituyen silenciosamente por otro sujeto."},
        {"key":p+"max_people_detected","label":"Máximo de sujetos detectados"+view_note,"value":info.get("max_people_detected"),"unit":"personas","quality":"Directa","notes":"Máximo número de personas con pose utilizable observado durante el seguimiento."},
    ]


def point_angle(ax, ay, bx, by, cx, cy):
    ba = np.array([ax - bx, ay - by], dtype=float)
    bc = np.array([cx - bx, cy - by], dtype=float)
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    c = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def robust_rom(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) >= 5 else np.nan


def rolling_smooth(arr, window=7):
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def zero_crossings(signal):
    s = np.asarray(signal, dtype=float)
    out = []
    for i in range(1, len(s)):
        if not (np.isfinite(s[i-1]) and np.isfinite(s[i])):
            continue
        if (s[i-1] <= 0 < s[i]) or (s[i-1] >= 0 > s[i]):
            out.append(i)
    return np.asarray(out, dtype=int)


def quality_label(score):
    return "Alta" if score >= 0.80 else ("Moderada" if score >= 0.65 else "Baja")


def add_angle_columns(seg):
    seg = seg.copy()
    for side in ("L", "R"):
        knee, hip, ankle, shoulder = [], [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            ha = point_angle(r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"])
            aa = point_angle(r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"])
            sa = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Elbow_x"], r[f"{side}Elbow_y"])
            knee.append(180.0 - ka if np.isfinite(ka) else np.nan)
            hip.append(180.0 - ha if np.isfinite(ha) else np.nan)
            ankle.append(aa)
            shoulder.append(sa)
        seg[f"{side}_knee_flex"] = knee
        seg[f"{side}_hip_flex"] = hip
        seg[f"{side}_ankle_angle"] = ankle
        seg[f"{side}_shoulder_elev"] = shoulder
    return seg


def axis_angle_to_vertical(x1, y1, x2, y2):
    """Ángulo firmado de un eje 2D respecto a la vertical de la imagen, plegado a [-90, 90]."""
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dx, -dy)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def axis_angle_to_horizontal(x1, y1, x2, y2):
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dy, dx)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def add_frontal_columns(seg):
    """Métricas proyectadas para vista frontal/posterior. No equivalen a rotaciones 3D ni a pronación clínica."""
    seg = seg.copy()
    for side in ("L", "R"):
        knee_dev, foot_prog, rearfoot = [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            knee_dev.append(abs(180.0-ka) if np.isfinite(ka) else np.nan)
            foot_prog.append(axis_angle_to_vertical(r[f"{side}Heel_x"], r[f"{side}Heel_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"]))
            rearfoot.append(axis_angle_to_vertical(r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}Heel_x"], r[f"{side}Heel_y"]))
        seg[f"{side}_frontal_knee_dev"] = knee_dev
        seg[f"{side}_foot_progress_proj"] = foot_prog
        seg[f"{side}_rearfoot_tilt_proj"] = rearfoot
    seg["pelvis_obliquity"] = [axis_angle_to_horizontal(r.LHip_x, r.LHip_y, r.RHip_x, r.RHip_y) for _, r in seg.iterrows()]
    seg["shoulder_obliquity"] = [axis_angle_to_horizontal(r.LShoulder_x, r.LShoulder_y, r.RShoulder_x, r.RShoulder_y) for _, r in seg.iterrows()]
    # Tronco en plano frontal/posterior: eje entre el centro pélvico y el centro de hombros.
    trunk_lean = []
    for _, r in seg.iterrows():
        hip_x = (r.LHip_x + r.RHip_x) / 2.0
        hip_y = (r.LHip_y + r.RHip_y) / 2.0
        sh_x = (r.LShoulder_x + r.RShoulder_x) / 2.0
        sh_y = (r.LShoulder_y + r.RShoulder_y) / 2.0
        trunk_lean.append(axis_angle_to_vertical(hip_x, hip_y, sh_x, sh_y))
    seg["trunk_lateral_lean"] = trunk_lean
    seg["shoulder_pelvis_rel"] = seg["shoulder_obliquity"] - seg["pelvis_obliquity"]
    pelvis_w = np.abs(seg.RHip_x.to_numpy(float) - seg.LHip_x.to_numpy(float))
    ankle_w = np.abs(seg.RAnkle_x.to_numpy(float) - seg.LAnkle_x.to_numpy(float))
    seg["base_width_relative"] = np.divide(ankle_w, pelvis_w, out=np.full_like(ankle_w, np.nan), where=pelvis_w>1e-6)
    return seg


def _foot_centroid(seg, side):
    x = (seg[f"{side}Ankle_x"].to_numpy(float) + seg[f"{side}Heel_x"].to_numpy(float) + seg[f"{side}BigToe_x"].to_numpy(float)) / 3.0
    y = (seg[f"{side}Ankle_y"].to_numpy(float) + seg[f"{side}Heel_y"].to_numpy(float) + seg[f"{side}BigToe_y"].to_numpy(float)) / 3.0
    return x, y


def _bool_runs(mask):
    """Devuelve runs True como (inicio, fin_exclusivo)."""
    m = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if start is not None and ((not v) or i == len(m) - 1):
            end = i if not v else i + 1
            runs.append((int(start), int(end)))
            start = None
    return runs


def _fill_short_false_gaps(mask, max_gap_frames, speed=None, speed_ceiling=None):
    m = np.asarray(mask, dtype=bool).copy()
    if len(m) == 0 or max_gap_frames <= 0:
        return m
    for a, b in _bool_runs(~m):
        if a == 0 or b == len(m):
            continue
        if (b - a) <= max_gap_frames:
            if speed is None or speed_ceiling is None:
                m[a:b] = True
            else:
                g = np.asarray(speed[a:b], float)
                if len(g) and np.isfinite(g).any() and float(np.nanmedian(g)) <= float(speed_ceiling):
                    m[a:b] = True
    return m


def _remove_short_true_runs(mask, min_run_frames):
    m = np.asarray(mask, dtype=bool).copy()
    for a, b in _bool_runs(m):
        if (b - a) < min_run_frames:
            m[a:b] = False
    return m


def _support_mask_2d(seg, side, fps, expected_stride_s=np.nan):
    """
    v0.9.2: estimación cinemática 2D de apoyo con continuidad temporal.
    Evita fragmentar un apoyo real en microbloques por jitter del tracking.

    Sigue siendo markerless 2D: NO sustituye footswitch, plataforma de fuerzas
    ni presión plantar.
    """
    x, y = _foot_centroid(seg, side)
    xs = (
        pd.Series(x).interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1).median()
        .rolling(5, center=True, min_periods=1).mean()
        .to_numpy(float)
    )
    ys = (
        pd.Series(y).interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1).median()
        .rolling(5, center=True, min_periods=1).mean()
        .to_numpy(float)
    )
    speed_px_s = np.hypot(np.gradient(xs), np.gradient(ys)) * float(fps)

    hip_x = (seg.LHip_x.to_numpy(float) + seg.RHip_x.to_numpy(float)) / 2.0
    hip_y = (seg.LHip_y.to_numpy(float) + seg.RHip_y.to_numpy(float)) / 2.0
    sh_x = (seg.LShoulder_x.to_numpy(float) + seg.RShoulder_x.to_numpy(float)) / 2.0
    sh_y = (seg.LShoulder_y.to_numpy(float) + seg.RShoulder_y.to_numpy(float)) / 2.0
    torso = np.hypot(sh_x - hip_x, sh_y - hip_y)
    torso = pd.Series(torso).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").to_numpy(float)
    med_torso = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(med_torso) or med_torso < 5:
        med_torso = 1.0
    speed = speed_px_s / med_torso

    finite = speed[np.isfinite(speed)]
    if len(finite) < max(20, int(fps)):
        return np.zeros(len(seg), dtype=bool), speed, {
            "quality": "No fiable", "score": 0.0, "reason": "Señal de pie insuficiente."
        }

    low = float(np.nanpercentile(finite, 38))
    high = float(np.nanpercentile(finite, 72))
    if high <= low:
        high = low + max(float(np.nanstd(finite)) * 0.35, 1e-6)

    state_support = bool(speed[0] <= (low + high) / 2.0) if np.isfinite(speed[0]) else True
    mask = np.zeros(len(speed), dtype=bool)
    for i, s in enumerate(speed):
        if not np.isfinite(s):
            mask[i] = state_support
            continue
        if state_support and s > high:
            state_support = False
        elif (not state_support) and s < low:
            state_support = True
        mask[i] = state_support

    if np.isfinite(expected_stride_s) and expected_stride_s > 0:
        gap_s = min(0.40, max(0.12, 0.09 * float(expected_stride_s)))
        min_support_s = min(0.80, max(0.18, 0.14 * float(expected_stride_s)))
        min_swing_s = min(0.50, max(0.12, 0.07 * float(expected_stride_s)))
    else:
        gap_s, min_support_s, min_swing_s = 0.20, 0.20, 0.12

    mask = _fill_short_false_gaps(
        mask,
        max_gap_frames=max(1, int(round(gap_s * fps))),
        speed=speed,
        speed_ceiling=high * 1.10,
    )
    mask = _remove_short_true_runs(mask, max(1, int(round(min_support_s * fps))))

    swing = ~mask
    swing = _remove_short_true_runs(swing, max(1, int(round(min_swing_s * fps))))
    mask = ~swing
    mask = _remove_short_true_runs(mask, max(1, int(round(min_support_s * fps))))

    return mask.astype(bool), speed, {
        "quality": "Pendiente", "score": np.nan,
        "reason": "Máscara estabilizada por histéresis y continuidad temporal.",
        "low_thr": low, "high_thr": high,
    }


def _support_cycle_summary(mask, fps, expected_stride_s=np.nan):
    """
    Convierte una máscara continua en ciclos IC→TO→siguiente IC y puntúa
    su consistencia. Los límites son controles matemáticos amplios, no
    rangos clínicos de normalidad.
    """
    starts, ends = _edges(mask)
    stance, swing, cycles, stance_pct = [], [], [], []
    valid_ic, valid_to, valid_next_ic = [], [], []

    for j in range(len(starts) - 1):
        ic = int(starts[j])
        nxt = int(starts[j + 1])
        offs = ends[(ends > ic) & (ends <= nxt)]
        if not len(offs):
            continue
        to = int(offs[0])
        cyc = (nxt - ic) / float(fps)
        st = (to - ic) / float(fps)
        sw = (nxt - to) / float(fps)
        if cyc <= 0 or st <= 0 or sw < 0:
            continue
        ratio = st / cyc
        if ratio < 0.10 or ratio > 0.985:
            continue
        if np.isfinite(expected_stride_s) and expected_stride_s > 0:
            if cyc < 0.45 * expected_stride_s or cyc > 1.80 * expected_stride_s:
                continue

        stance.append(st)
        swing.append(sw)
        cycles.append(cyc)
        stance_pct.append(ratio * 100.0)
        valid_ic.append(ic)
        valid_to.append(to)
        valid_next_ic.append(nxt)

    stance = np.asarray(stance, float)
    swing = np.asarray(swing, float)
    cycles = np.asarray(cycles, float)
    stance_pct = np.asarray(stance_pct, float)

    score = 100.0
    reasons = []

    if len(cycles) < 2:
        score -= 55
        reasons.append("menos de 2 ciclos completos")
    elif len(cycles) < 3:
        score -= 15
        reasons.append("pocos ciclos completos")

    err = np.nan
    if len(cycles) and np.isfinite(expected_stride_s) and expected_stride_s > 0:
        cyc_med = float(np.nanmedian(cycles))
        err = abs(cyc_med - expected_stride_s) / expected_stride_s * 100.0
        if err > 50:
            score -= 45
            reasons.append("periodo de ciclo discordante con la cadencia")
        elif err > 30:
            score -= 25
            reasons.append("periodo de ciclo moderadamente discordante")
        elif err > 18:
            score -= 10
            reasons.append("periodo de ciclo con ligera discordancia")

    fragmentation = np.nan
    if np.isfinite(expected_stride_s) and expected_stride_s > 0 and len(mask) > 0:
        duration = len(mask) / float(fps)
        expected_same_side_cycles = max(duration / expected_stride_s, 1.0)
        fragmentation = len(starts) / expected_same_side_cycles
        if fragmentation > 2.2:
            score -= 35
            reasons.append("segmentación fragmentada")
        elif fragmentation > 1.6:
            score -= 15
            reasons.append("posible fragmentación residual")

    score = float(max(0.0, min(100.0, score)))
    quality = "Alta" if score >= 75 else ("Moderada" if score >= 55 else "No fiable")

    return {
        "stance": stance,
        "swing": swing,
        "cycle": cycles,
        "stance_pct": stance_pct,
        "swing_pct": 100.0 - stance_pct if len(stance_pct) else np.asarray([], float),
        "ic": np.asarray(valid_ic, int),
        "to": np.asarray(valid_to, int),
        "next_ic": np.asarray(valid_next_ic, int),
        "starts": np.asarray(starts, int),
        "ends": np.asarray(ends, int),
        "quality": quality,
        "score": score,
        "cycle_error_pct": float(err) if np.isfinite(err) else np.nan,
        "fragmentation_index": float(fragmentation) if np.isfinite(fragmentation) else np.nan,
        "reason": ", ".join(reasons) if reasons else "consistencia temporal suficiente",
    }



def _automatic_straight_walking_mask(seg, fps):
    """
    v0.9.2: detecta y excluye automáticamente transiciones/giro de 180° en
    registros frontal/posterior.

    Combina dos señales normalizadas:
      1) estrechamiento transitorio del ancho de hombros/caderas respecto al
         tronco (el cuerpo se pone de perfil durante el giro);
      2) inversión sostenida de la tendencia de tamaño corporal aparente
         (alejarse -> acercarse, o viceversa).

    Es un filtro de calidad, no un detector clínico de giro.
    Si la evidencia no es suficiente, solo recorta bordes del registro.
    """
    n = len(seg)
    if n == 0:
        return np.zeros(0, dtype=bool), {
            "turn_detected": False, "excluded_pct": np.nan,
            "reason": "segmento vacío"
        }

    fps = float(max(fps, 1.0))
    edge = min(max(1, int(round(0.35 * fps))), max(1, n // 8))
    valid = np.ones(n, dtype=bool)
    if n > 2 * edge:
        valid[:edge] = False
        valid[-edge:] = False

    lx = seg.LShoulder_x.to_numpy(float); rx = seg.RShoulder_x.to_numpy(float)
    ly = seg.LShoulder_y.to_numpy(float); ry = seg.RShoulder_y.to_numpy(float)
    lhx = seg.LHip_x.to_numpy(float); rhx = seg.RHip_x.to_numpy(float)
    lhy = seg.LHip_y.to_numpy(float); rhy = seg.RHip_y.to_numpy(float)

    shoulder_w = np.abs(rx - lx)
    hip_w = np.abs(rhx - lhx)
    shoulder_y = (ly + ry) / 2.0
    hip_y = (lhy + rhy) / 2.0
    torso_h = np.abs(hip_y - shoulder_y)

    def _smooth(a, w):
        return (
            pd.Series(a).replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
            .rolling(w, center=True, min_periods=1).median()
            .rolling(w, center=True, min_periods=1).mean()
            .to_numpy(float)
        )

    w = max(5, int(round(0.25 * fps)) | 1)
    shoulder_w = _smooth(shoulder_w, w)
    hip_w = _smooth(hip_w, w)
    torso_h = _smooth(torso_h, w)

    denom = np.maximum(torso_h, 1e-6)
    frontal_ratio = 0.65 * (shoulder_w / denom) + 0.35 * (hip_w / denom)
    finite_ratio = frontal_ratio[np.isfinite(frontal_ratio)]
    turn_candidates = np.zeros(n, dtype=bool)

    # Evidencia 1: orientación transitoria de perfil.
    if len(finite_ratio) >= max(20, int(fps)):
        reference = float(np.nanpercentile(finite_ratio, 75))
        if np.isfinite(reference) and reference > 1e-6:
            turn_candidates |= frontal_ratio < (0.62 * reference)

    # Evidencia 2: inversión sostenida del tamaño aparente del cuerpo.
    # Evita declarar giro por un único frame: compara tendencias a ambos lados.
    scale = _smooth(torso_h, max(7, int(round(0.40 * fps)) | 1))
    look = max(5, int(round(0.85 * fps)))
    reversal_scores = np.zeros(n, dtype=float)
    for i in range(look, n - look):
        pre = scale[i] - scale[i - look]
        post = scale[i + look] - scale[i]
        local = float(np.nanmedian(scale[max(0, i-look):min(n, i+look+1)]))
        if not np.isfinite(local) or local <= 1e-6:
            continue
        # signo opuesto y cambio conjunto de al menos ~5 % del tamaño local
        if np.isfinite(pre) and np.isfinite(post) and pre * post < 0:
            reversal_scores[i] = (abs(pre) + abs(post)) / local

    if np.nanmax(reversal_scores) >= 0.05:
        i0 = int(np.nanargmax(reversal_scores))
        turn_candidates[i0] = True

    # Agrupa el giro y añade margen temporal para no contaminar ciclos vecinos.
    turn_detected = bool(turn_candidates.any())
    if turn_detected:
        idx = np.where(turn_candidates)[0]
        # Si hay muchos candidatos, conservar el cluster principal alrededor
        # del punto con menor frontalidad / mayor inversión.
        center = int(np.median(idx))
        if np.nanmax(reversal_scores) >= 0.05:
            center = int(np.nanargmax(reversal_scores))
        pad = max(1, int(round(0.75 * fps)))
        a = max(0, center - pad)
        b = min(n, center + pad + 1)
        valid[a:b] = False

    # No aplicar una exclusión automática agresiva si deja poco material útil.
    min_keep = max(int(round(3.0 * fps)), int(round(0.45 * n)))
    if valid.sum() < min_keep:
        valid[:] = True
        if n > 2 * edge:
            valid[:edge] = False
            valid[-edge:] = False
        turn_detected = False
        reason = "evidencia de giro no suficientemente robusta; solo se recortan bordes"
    else:
        reason = "giro/transición excluido automáticamente" if turn_detected else "sin giro robusto detectado; bordes excluidos"

    return valid, {
        "turn_detected": turn_detected,
        "excluded_pct": float((1.0 - valid.mean()) * 100.0),
        "usable_pct": float(valid.mean() * 100.0),
        "reason": reason,
    }


def _filter_support_summary_to_mask(summary, valid_mask, fps, min_fraction=0.90):
    """
    Conserva únicamente ciclos IC→TO→siguiente IC contenidos casi por completo
    en el dominio rectilíneo válido.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    keys = ["stance", "swing", "cycle", "stance_pct", "swing_pct", "ic", "to", "next_ic"]
    arrs = {k: np.asarray(summary.get(k, [])) for k in keys}

    ncyc = len(arrs["ic"])
    keep = []
    for j in range(ncyc):
        ic = int(arrs["ic"][j])
        nxt = int(arrs["next_ic"][j]) if j < len(arrs["next_ic"]) else int(round(ic + arrs["cycle"][j] * fps))
        a, b = max(0, ic), min(len(valid_mask), max(ic + 1, nxt))
        frac = float(valid_mask[a:b].mean()) if b > a else 0.0
        keep.append(frac >= float(min_fraction))
    keep = np.asarray(keep, dtype=bool)

    out = dict(summary)
    for k in keys:
        if len(arrs[k]) == ncyc:
            out[k] = arrs[k][keep]

    # Cobertura temporal de ciclos aceptados.
    coverage = np.zeros(len(valid_mask), dtype=bool)
    for ic, nxt in zip(np.asarray(out.get("ic", []), int), np.asarray(out.get("next_ic", []), int)):
        coverage[max(0, ic):min(len(coverage), max(ic + 1, nxt))] = True
    out["coverage_mask"] = coverage & valid_mask

    # Reevalúa la calidad si la exclusión deja muy pocos ciclos.
    n = len(out.get("cycle", []))
    score = float(summary.get("score", 0.0))
    reasons = [str(summary.get("reason", ""))]
    if n < 2:
        score = min(score, 45.0)
        reasons.append("menos de 2 ciclos rectilíneos válidos")
    elif n < 3:
        score = min(score, 70.0)
        reasons.append("solo 2 ciclos rectilíneos válidos")
    quality = "Alta" if score >= 75 else ("Moderada" if score >= 55 else "No fiable")
    out["score"] = score
    out["quality"] = quality
    out["reason"] = ", ".join([r for r in reasons if r])
    return out


def _contact_step_metrics(L, R, fps):
    """
    Cadencia, CV y asimetría a partir de contactos iniciales I/D de ciclos
    validados. Evita usar cruces geométricos pares/impares como sustituto de lado.
    """
    events = []
    for i in np.asarray(L.get("ic", []), int):
        events.append((int(i), "L"))
    for i in np.asarray(R.get("ic", []), int):
        events.append((int(i), "R"))
    events.sort(key=lambda z: z[0])

    # Fusiona duplicados muy próximos del mismo evento y exige alternancia I/D.
    cleaned = []
    min_sep = max(1, int(round(0.18 * float(fps))))
    for idx, side in events:
        if cleaned and idx - cleaned[-1][0] < min_sep:
            # En caso de conflicto conservar el evento que mantiene alternancia.
            if side != cleaned[-1][1] and len(cleaned) >= 2 and side != cleaned[-2][1]:
                cleaned[-1] = (idx, side)
            continue
        if cleaned and side == cleaned[-1][1]:
            continue
        cleaned.append((idx, side))

    if len(cleaned) < 4:
        return {
            "cadence": np.nan, "mean_step": np.nan, "cv": np.nan, "asym": np.nan,
            "events": cleaned, "n_intervals": 0, "quality": "No fiable",
            "reason": "menos de 4 contactos alternantes válidos"
        }

    frames = np.asarray([e[0] for e in cleaned], float)
    sides = [e[1] for e in cleaned]
    intervals = np.diff(frames) / float(fps)
    med = float(np.nanmedian(intervals))

    # Solo elimina intervalos manifiestamente incompatibles con un paso;
    # no recorta variabilidad clínica moderada.
    good = np.isfinite(intervals) & (intervals >= max(0.18, 0.45 * med)) & (intervals <= min(2.0, 1.80 * med))
    intervals2 = intervals[good]
    transitions = [(sides[i], sides[i+1]) for i in range(len(intervals)) if good[i]]

    if len(intervals2) < 3 or np.nanmean(intervals2) <= 0:
        return {
            "cadence": np.nan, "mean_step": np.nan, "cv": np.nan, "asym": np.nan,
            "events": cleaned, "n_intervals": int(len(intervals2)), "quality": "No fiable",
            "reason": "insuficientes intervalos de paso plausibles"
        }

    mean_step = float(np.nanmean(intervals2))
    cadence = float(60.0 / mean_step)
    cv = float(np.nanstd(intervals2, ddof=1) / mean_step * 100.0) if len(intervals2) >= 3 else np.nan

    lr = [v for v, tr in zip(intervals2, transitions) if tr == ("L", "R")]
    rl = [v for v, tr in zip(intervals2, transitions) if tr == ("R", "L")]
    if len(lr) >= 2 and len(rl) >= 2:
        ml, mr = float(np.mean(lr)), float(np.mean(rl))
        asym = abs(ml - mr) / ((ml + mr) / 2.0) * 100.0 if (ml + mr) > 0 else np.nan
    else:
        asym = np.nan

    quality = "Alta" if len(intervals2) >= 6 else "Moderada"
    return {
        "cadence": cadence, "mean_step": mean_step, "cv": cv, "asym": asym,
        "events": cleaned, "n_intervals": int(len(intervals2)), "quality": quality,
        "reason": f"{len(intervals2)} intervalos derivados de contactos I/D validados"
    }





def _kinematic_alternation_metrics(seg, fps):
    """
    v0.10.6 · Ritmo independiente de las máscaras de apoyo.

    Usa exclusivamente trayectorias articulares distales ya trackeadas:
      señal = posición vertical relativa pie derecho - pie izquierdo
    normalizada por la altura aparente del tronco y detrendida lentamente.

    Los máximos/mínimos alternantes representan posiciones relativas sucesivas
    de avance de una y otra extremidad. No equivalen a footswitch/GRF, pero son
    apropiados como detector cinemático de periodicidad cuando el tracking es
    bueno y las máscaras de contacto fallan.

    Devuelve:
      - cadence: pasos/min
      - asym: asimetría temporal de alternancia (%)
      - cv: CV robusto del intervalo de alternancia (%)
      - events: posiciones dentro de seg
      - event_frames: números de frame reales
    """
    if seg is None or len(seg) < max(45, int(round(2.5*fps))):
        return {
            "cadence": np.nan, "asym": np.nan, "cv": np.nan,
            "events": np.asarray([], int), "event_frames": np.asarray([], int),
            "intervals": np.asarray([], float), "n_intervals": 0,
            "quality": "No fiable", "periodicity": np.nan,
            "reason": "segmento insuficiente"
        }

    try:
        ly = (
            pd.to_numeric(seg["LAnkle_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["LHeel_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["LBigToe_y"], errors="coerce").to_numpy(float)
        ) / 3.0
        ry = (
            pd.to_numeric(seg["RAnkle_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RHeel_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RBigToe_y"], errors="coerce").to_numpy(float)
        ) / 3.0

        hip_y = (
            pd.to_numeric(seg["LHip_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RHip_y"], errors="coerce").to_numpy(float)
        ) / 2.0
        sh_y = (
            pd.to_numeric(seg["LShoulder_y"], errors="coerce").to_numpy(float)
            + pd.to_numeric(seg["RShoulder_y"], errors="coerce").to_numpy(float)
        ) / 2.0
        torso = np.abs(hip_y-sh_y)
        torso_med = float(np.nanmedian(torso[np.isfinite(torso) & (torso > 5)])) if np.any(np.isfinite(torso) & (torso > 5)) else np.nan
        denom = np.where(np.isfinite(torso) & (torso > 5), torso, torso_med if np.isfinite(torso_med) else 1.0)

        sig = (ry-ly) / denom
        sig = (
            pd.Series(sig)
            .interpolate(limit_direction="both")
            .rolling(max(3, int(round(0.22*fps))) | 1, center=True, min_periods=1)
            .mean()
            .to_numpy(float)
        )

        # Eliminar deriva por perspectiva/aproximación a cámara sin borrar pasos lentos.
        trend_w = max(15, int(round(2.2*fps)))
        if trend_w % 2 == 0:
            trend_w += 1
        trend = pd.Series(sig).rolling(trend_w, center=True, min_periods=1).mean().to_numpy(float)
        z = sig - trend

        amp = float(np.nanstd(z))
        if not np.isfinite(amp) or amp < 0.008:
            return {
                "cadence": np.nan, "asym": np.nan, "cv": np.nan,
                "events": np.asarray([], int), "event_frames": np.asarray([], int),
                "intervals": np.asarray([], float), "n_intervals": 0,
                "quality": "No fiable", "periodicity": np.nan,
                "reason": "amplitud de alternancia insuficiente"
            }

        # Un pico del mismo signo no puede repetirse demasiado pronto.
        distance = max(3, int(round(0.40*fps)))
        prominence = max(0.012, 0.34*amp)

        peaks, pprop = find_peaks(z, distance=distance, prominence=prominence)
        troughs, tprop = find_peaks(-z, distance=distance, prominence=prominence)

        candidates = []
        for k, i in enumerate(peaks):
            candidates.append((int(i), 1, float(pprop["prominences"][k])))
        for k, i in enumerate(troughs):
            candidates.append((int(i), -1, float(tprop["prominences"][k])))
        candidates.sort(key=lambda e: e[0])

        # Exigir alternancia de signo. Si aparecen dos extremos iguales seguidos,
        # conservar el más prominente.
        events = []
        for ev in candidates:
            if not events:
                events.append(ev)
                continue
            if ev[1] == events[-1][1]:
                if ev[2] > events[-1][2]:
                    events[-1] = ev
            else:
                events.append(ev)

        if len(events) < 5:
            return {
                "cadence": np.nan, "asym": np.nan, "cv": np.nan,
                "events": np.asarray([e[0] for e in events], int),
                "event_frames": np.asarray([], int),
                "intervals": np.asarray([], float), "n_intervals": max(0,len(events)-1),
                "quality": "No fiable", "periodicity": np.nan,
                "reason": "menos de 5 extremos alternantes"
            }

        pos = np.asarray([e[0] for e in events], int)
        dirs = np.asarray([e[1] for e in events], int)
        dt = np.diff(pos) / float(fps)
        trans_dir = dirs[:-1]  # +1→-1 o -1→+1

        # Límites muy amplios para marcha neurológica lenta.
        plausible = np.isfinite(dt) & (dt >= 0.30) & (dt <= 2.50)
        dt0 = dt[plausible]
        td0 = trans_dir[plausible]

        if len(dt0) < 4:
            return {
                "cadence": np.nan, "asym": np.nan, "cv": np.nan,
                "events": pos, "event_frames": np.asarray([], int),
                "intervals": dt0, "n_intervals": int(len(dt0)),
                "quality": "No fiable", "periodicity": np.nan,
                "reason": "menos de 4 intervalos de alternancia plausibles"
            }

        # Rechazo robusto suave: preserva variabilidad patológica real.
        med = float(np.nanmedian(dt0))
        mad = float(np.nanmedian(np.abs(dt0-med)))
        keep = np.ones(len(dt0), dtype=bool)
        if len(dt0) >= 7 and mad > 1e-6:
            rz = 0.6745*np.abs(dt0-med)/mad
            keep &= rz <= 4.5
        keep &= (dt0 >= max(0.30, 0.45*med)) & (dt0 <= min(2.50, 2.20*med))

        dtk = dt0[keep]
        tdk = td0[keep]
        if len(dtk) < 4:
            dtk, tdk = dt0, td0

        # Mediana = cadencia robusta frente a una pausa o paso muy prolongado.
        step_med = float(np.nanmedian(dtk))
        cadence = float(60.0/step_med) if step_med > 0 else np.nan

        # CV robusto con SD clásica sobre intervalos ya depurados.
        cv = (
            float(np.nanstd(dtk, ddof=1)/np.nanmean(dtk)*100.0)
            if len(dtk) >= 3 and np.nanmean(dtk) > 0 else np.nan
        )

        a = dtk[tdk > 0]
        b = dtk[tdk < 0]
        asym = np.nan
        if len(a) >= 2 and len(b) >= 2:
            ma, mb = float(np.nanmedian(a)), float(np.nanmedian(b))
            den = (ma+mb)/2.0
            asym = abs(ma-mb)/den*100.0 if den > 0 else np.nan

        # Periodicidad independiente mediante autocorrelación: pico de zancada.
        zz = np.nan_to_num(z - np.nanmean(z))
        ac = np.correlate(zz, zz, mode="full")[len(zz)-1:]
        periodicity = np.nan
        if len(ac) and ac[0] > 1e-12:
            ac = ac/ac[0]
            lo = max(2, int(round(0.65*fps)))
            hi = min(len(ac)-2, int(round(5.0*fps)))
            loc = []
            for i in range(lo+1, hi):
                if ac[i] > ac[i-1] and ac[i] >= ac[i+1]:
                    loc.append(i)
            if loc:
                best = max(loc, key=lambda i: ac[i])
                periodicity = float(ac[best])

        nint = int(len(dtk))
        if nint >= 10 and (not np.isfinite(periodicity) or periodicity >= 0.10):
            quality = "Alta"
        elif nint >= 6:
            quality = "Moderada"
        else:
            quality = "Baja"

        frame_vals = pd.to_numeric(seg["frame"], errors="coerce").to_numpy()
        event_frames = []
        for p in pos:
            if 0 <= p < len(frame_vals) and np.isfinite(frame_vals[p]):
                event_frames.append(int(frame_vals[p]))

        return {
            "cadence": cadence,
            "asym": asym,
            "cv": cv,
            "events": pos,
            "event_frames": np.asarray(event_frames, int),
            "intervals": dtk,
            "n_intervals": nint,
            "quality": quality,
            "periodicity": periodicity,
            "reason": f"{nint} intervalos de alternancia distal válidos"
        }
    except Exception as e:
        return {
            "cadence": np.nan, "asym": np.nan, "cv": np.nan,
            "events": np.asarray([], int), "event_frames": np.asarray([], int),
            "intervals": np.asarray([], float), "n_intervals": 0,
            "quality": "No fiable", "periodicity": np.nan,
            "reason": f"error detector cinemático: {e}"
        }


def _canonical_gait_timeline(L, R, fps):
    """
    v0.9.2 · Línea temporal anatómica canónica.

    Construye una única secuencia temporal de contactos iniciales (IC)
    izquierdos/derechos procedentes de LOS MISMOS ciclos IC→TO→IC usados
    para apoyo/oscilación.

    Principios:
    - exige alternancia anatómica L-R-L-R;
    - no usa pares/impares como sustituto de lateralidad;
    - prioriza intervalos próximos a 1/2 del ciclo ipsilateral mediano;
    - conserva la secuencia temporal de mayor coherencia;
    - no elimina un ciclo únicamente porque aumente el CV.
    """
    fps = float(max(fps, 1.0))
    events = []
    for i in np.asarray(L.get("ic", []), dtype=int):
        events.append((int(i), "L"))
    for i in np.asarray(R.get("ic", []), dtype=int):
        events.append((int(i), "R"))
    events.sort(key=lambda z: (z[0], z[1]))

    raw_cycles = np.r_[
        np.asarray(L.get("cycle", []), dtype=float),
        np.asarray(R.get("cycle", []), dtype=float)
    ]
    raw_cycles = raw_cycles[np.isfinite(raw_cycles) & (raw_cycles >= 0.55) & (raw_cycles <= 2.50)]
    stride_ref = float(np.nanmedian(raw_cycles)) if len(raw_cycles) else np.nan
    step_ref = stride_ref / 2.0 if np.isfinite(stride_ref) and stride_ref > 0 else np.nan

    if len(events) < 4:
        return {
            "events": events, "intervals": np.asarray([], float),
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": 0, "quality": "No fiable",
            "reason": "menos de 4 IC anatómicos disponibles",
            "stride_ref": stride_ref,
        }

    # Elimina duplicados del mismo lado extremadamente próximos.
    dedup = []
    duplicate_gap = max(1, int(round(0.16 * fps)))
    for ev in events:
        if dedup and ev[1] == dedup[-1][1] and ev[0] - dedup[-1][0] < duplicate_gap:
            continue
        dedup.append(ev)
    events = dedup

    # Intervalo de paso plausible. Es amplio para no borrar variabilidad clínica,
    # pero evita saltos manifiestamente incompatibles con un paso.
    if np.isfinite(step_ref):
        lo_step = max(0.22, 0.58 * step_ref)
        hi_step = min(1.35, 1.55 * step_ref)
    else:
        lo_step, hi_step = 0.22, 1.35

    # Programación dinámica: mejor cadena alternante.
    # score alto = más eventos + menor desviación respecto al paso de referencia.
    n = len(events)
    best_score = np.ones(n, dtype=float)
    prev_idx = np.full(n, -1, dtype=int)
    chain_len = np.ones(n, dtype=int)

    for j in range(n):
        fj, sj = events[j]
        for i in range(j):
            fi, si = events[i]
            if si == sj:
                continue
            dt = (fj - fi) / fps
            if not (lo_step <= dt <= hi_step):
                continue
            penalty = 0.0
            if np.isfinite(step_ref) and step_ref > 0:
                penalty = min(abs(dt - step_ref) / step_ref, 1.0)
            candidate = best_score[i] + 1.0 - 0.18 * penalty
            if candidate > best_score[j]:
                best_score[j] = candidate
                prev_idx[j] = i
                chain_len[j] = chain_len[i] + 1

    end = int(np.argmax(best_score + 0.02 * chain_len))
    idxs = []
    k = end
    while k >= 0:
        idxs.append(k)
        k = int(prev_idx[k])
    idxs = idxs[::-1]
    chain = [events[i] for i in idxs]

    if len(chain) < 4:
        return {
            "events": chain, "intervals": np.asarray([], float),
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": max(0, len(chain)-1), "quality": "No fiable",
            "reason": "no se obtuvo una cadena L-R alternante suficientemente larga",
            "stride_ref": stride_ref,
        }

    frames = np.asarray([e[0] for e in chain], dtype=float)
    sides = [e[1] for e in chain]
    intervals = np.diff(frames) / fps
    transitions = [(sides[i], sides[i+1]) for i in range(len(sides)-1)]

    good = np.isfinite(intervals) & (intervals >= lo_step) & (intervals <= hi_step)
    intervals_good = intervals[good]
    trans_good = [tr for tr, ok in zip(transitions, good) if ok]

    if len(intervals_good) < 3 or float(np.nanmean(intervals_good)) <= 0:
        return {
            "events": chain, "intervals": intervals_good,
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": int(len(intervals_good)), "quality": "No fiable",
            "reason": "menos de 3 intervalos de paso anatómicos plausibles",
            "stride_ref": stride_ref,
        }

    mean_step = float(np.nanmean(intervals_good))
    cadence = float(60.0 / mean_step)

    lr = np.asarray([v for v, tr in zip(intervals_good, trans_good) if tr == ("L","R")], float)
    rl = np.asarray([v for v, tr in zip(intervals_good, trans_good) if tr == ("R","L")], float)
    lr_mean = float(np.nanmean(lr)) if len(lr) else np.nan
    rl_mean = float(np.nanmean(rl)) if len(rl) else np.nan

    if len(lr) >= 2 and len(rl) >= 2 and np.isfinite(lr_mean) and np.isfinite(rl_mean):
        denom = (lr_mean + rl_mean) / 2.0
        asym = abs(lr_mean - rl_mean) / denom * 100.0 if denom > 0 else np.nan
    else:
        asym = np.nan

    if len(intervals_good) >= 7 and len(lr) >= 3 and len(rl) >= 3:
        quality = "Alta"
    elif len(intervals_good) >= 5:
        quality = "Moderada"
    else:
        quality = "Baja"

    return {
        "events": chain,
        "intervals": intervals_good,
        "cadence": cadence,
        "mean_step": mean_step,
        "asym": asym,
        "lr_mean": lr_mean,
        "rl_mean": rl_mean,
        "n_lr": int(len(lr)),
        "n_rl": int(len(rl)),
        "n_intervals": int(len(intervals_good)),
        "quality": quality,
        "reason": f"{len(intervals_good)} pasos en cadena anatómica alternante",
        "stride_ref": stride_ref,
    }


def _raw_stride_cadence(L, R):
    """
    Cadencia de control desde ciclos ipsilaterales SIN usar el filtro del CV.
    Así un ciclo estadísticamente variable no altera silenciosamente el ritmo.
    """
    vals = np.r_[
        np.asarray(L.get("cycle", []), dtype=float),
        np.asarray(R.get("cycle", []), dtype=float)
    ]
    vals = vals[np.isfinite(vals) & (vals >= 0.55) & (vals <= 2.50)]
    if len(vals) < 2:
        return np.nan, np.nan, int(len(vals))
    mean_stride = float(np.nanmean(vals))
    cadence = float(120.0 / mean_stride) if mean_stride > 0 else np.nan
    return cadence, mean_stride, int(len(vals))


def _temporal_closure_from_masks(lmask, rmask, L, R, straight_mask, fps):
    """
    Control físico independiente v0.9.2.

    Para marcha sin fase aérea relevante:
        stance_L + stance_R = ciclo * (1 + doble_apoyo_fracción)

    Permite estimar qué cadencia sería compatible simultáneamente con
    tiempos de apoyo y solapamiento bipodal observados.
    """
    cov_l = np.asarray(L.get("coverage_mask", np.ones(len(lmask), dtype=bool)), bool)
    cov_r = np.asarray(R.get("coverage_mask", np.ones(len(rmask), dtype=bool)), bool)
    valid = cov_l & cov_r & np.asarray(straight_mask, bool)

    if not valid.any():
        return {
            "double_raw": np.nan, "flight_pct": np.nan,
            "stance_pct_l": np.nan, "stance_pct_r": np.nan,
            "double_expected": np.nan, "double_discrepancy": np.nan,
            "closure_stride": np.nan, "closure_cadence": np.nan,
        }

    both = np.asarray(lmask, bool) & np.asarray(rmask, bool)
    none = (~np.asarray(lmask, bool)) & (~np.asarray(rmask, bool))
    double_raw = float(np.mean(both[valid]) * 100.0)
    flight_pct = float(np.mean(none[valid]) * 100.0)

    st_l = float(np.nanmean(L.get("stance", []))) if len(L.get("stance", [])) else np.nan
    st_r = float(np.nanmean(R.get("stance", []))) if len(R.get("stance", [])) else np.nan
    sp_l = float(np.nanmean(L.get("stance_pct", []))) if len(L.get("stance_pct", [])) else np.nan
    sp_r = float(np.nanmean(R.get("stance_pct", []))) if len(R.get("stance_pct", [])) else np.nan

    double_expected = max(0.0, sp_l + sp_r - 100.0) if np.isfinite(sp_l) and np.isfinite(sp_r) else np.nan
    double_discrepancy = abs(double_raw - double_expected) if np.isfinite(double_raw) and np.isfinite(double_expected) else np.nan

    closure_stride = np.nan
    closure_cadence = np.nan
    if (
        np.isfinite(st_l) and np.isfinite(st_r) and
        np.isfinite(double_raw) and
        np.isfinite(flight_pct) and flight_pct <= 5.0
    ):
        denom = 1.0 + double_raw / 100.0
        if denom > 0:
            closure_stride = float((st_l + st_r) / denom)
            if closure_stride > 0:
                closure_cadence = float(120.0 / closure_stride)

    return {
        "double_raw": double_raw,
        "flight_pct": flight_pct,
        "stance_pct_l": sp_l,
        "stance_pct_r": sp_r,
        "double_expected": double_expected,
        "double_discrepancy": double_discrepancy,
        "closure_stride": closure_stride,
        "closure_cadence": closure_cadence,
    }


def _pct_disagreement(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return np.nan
    return float(abs(a-b) / ((a+b)/2.0) * 100.0)


def _cycle_timing_metrics(L, R, fps):
    """
    v0.9.2 · Estimador robusto de VARIABILIDAD por lado.

    Objetivos:
    - Estimar el CV sin mezclar directamente ambas piernas.
    - Evitar inflar el CV mezclando directamente ciclos izquierdos y derechos.
    - Rechazar ciclos atípicos de forma robusta por lado.
    - Informar tamaño muestral y confianza.
    """

    def _robust_side(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) == 0:
            return {
                "raw": x, "clean": x, "mean": np.nan, "sd": np.nan, "cv": np.nan,
                "n_raw": 0, "n_clean": 0, "outliers": 0
            }

        # Filtro fisiológicamente amplio inicial.
        x = x[(x >= 0.55) & (x <= 2.50)]
        if len(x) == 0:
            return {
                "raw": x, "clean": x, "mean": np.nan, "sd": np.nan, "cv": np.nan,
                "n_raw": 0, "n_clean": 0, "outliers": 0
            }

        raw = x.copy()
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))

        if len(x) >= 4 and mad > 1e-9:
            robust_z = 0.6745 * np.abs(x - med) / mad
            keep = robust_z <= 3.5
            x = x[keep]

        # Segundo filtro relativo conservador para evitar falsos IC→IC.
        if len(x) >= 3:
            med2 = float(np.nanmedian(x))
            lo = max(0.55, 0.75 * med2)
            hi = min(2.50, 1.25 * med2)
            x = x[(x >= lo) & (x <= hi)]

        mean = float(np.nanmean(x)) if len(x) else np.nan
        sd = float(np.nanstd(x, ddof=1)) if len(x) >= 2 else np.nan
        cv = float(sd / mean * 100.0) if np.isfinite(sd) and np.isfinite(mean) and mean > 0 else np.nan

        return {
            "raw": raw,
            "clean": x,
            "mean": mean,
            "sd": sd,
            "cv": cv,
            "n_raw": int(len(raw)),
            "n_clean": int(len(x)),
            "outliers": int(max(0, len(raw) - len(x))),
        }

    left = _robust_side(L.get("cycle", []))
    right = _robust_side(R.get("cycle", []))

    # Cadencia global desde la duración media de zancada por lado.
    side_means = [v for v in (left["mean"], right["mean"]) if np.isfinite(v)]
    if len(side_means) == 0:
        return {
            "cadence": np.nan, "mean_stride": np.nan, "cv": np.nan,
            "cv_left": np.nan, "cv_right": np.nan,
            "l_mean_stride": np.nan, "r_mean_stride": np.nan,
            "n_left": left["n_clean"], "n_right": right["n_clean"],
            "n_cycles": left["n_clean"] + right["n_clean"],
            "outliers_left": left["outliers"], "outliers_right": right["outliers"],
            "quality": "No fiable",
            "reason": "sin ciclos IC→IC válidos"
        }

    # Media ponderada por número de ciclos válidos.
    num = 0.0
    den = 0
    for side in (left, right):
        if np.isfinite(side["mean"]) and side["n_clean"] > 0:
            num += side["mean"] * side["n_clean"]
            den += side["n_clean"]
    mean_stride = float(num / den) if den > 0 else float(np.nanmean(side_means))
    cadence = float(120.0 / mean_stride) if mean_stride > 0 else np.nan

    # CV global robusto = promedio ponderado de CV izquierdo y derecho.
    cv_num = 0.0
    cv_den = 0
    for side in (left, right):
        if np.isfinite(side["cv"]) and side["n_clean"] >= 2:
            cv_num += side["cv"] * side["n_clean"]
            cv_den += side["n_clean"]
    cv_global = float(cv_num / cv_den) if cv_den > 0 else np.nan

    nL, nR = left["n_clean"], right["n_clean"]
    n_total = nL + nR

    # Confianza explícita según tamaño muestral bilateral.
    if nL >= 4 and nR >= 4:
        quality = "Alta"
    elif nL >= 3 and nR >= 3:
        quality = "Moderada"
    elif nL >= 2 and nR >= 2:
        quality = "Baja"
    else:
        quality = "Muy baja"

    return {
        "cadence": cadence,
        "mean_stride": mean_stride,
        "cv": cv_global,
        "cv_left": left["cv"],
        "cv_right": right["cv"],
        "l_mean_stride": left["mean"],
        "r_mean_stride": right["mean"],
        "n_left": nL,
        "n_right": nR,
        "n_cycles": n_total,
        "outliers_left": left["outliers"],
        "outliers_right": right["outliers"],
        "quality": quality,
        "reason": (
            f"{nL} ciclos izquierdos + {nR} derechos tras filtrado robusto; "
            f"outliers excluidos L={left['outliers']}, R={right['outliers']}"
        )
    }


def _run_durations(mask, fps, min_s=0.15, max_s=10.0):
    vals = []
    for a, b in _bool_runs(mask):
        d = (b - a) / float(fps)
        if min_s <= d <= max_s:
            vals.append(d)
    return np.asarray(vals, float)


def _medial_knee_deviation_angle(row, side):
    hx,hy=row[f"{side}Hip_x"],row[f"{side}Hip_y"]
    kx,ky=row[f"{side}Knee_x"],row[f"{side}Knee_y"]
    ax,ay=row[f"{side}Ankle_x"],row[f"{side}Ankle_y"]
    midx=(row.LHip_x+row.RHip_x)/2.0
    if not all(np.isfinite([hx,hy,kx,ky,ax,ay,midx])) or abs(ay-hy)<1e-6:
        return np.nan
    t=(ky-hy)/(ay-hy)
    line_x=hx+t*(ax-hx)
    toward_mid=(kx-line_x)*(midx-line_x)>0
    if not toward_mid:
        return 0.0
    ka=point_angle(hx,hy,kx,ky,ax,ay)
    return abs(180.0-ka) if np.isfinite(ka) else np.nan

def add_frontal_advanced(seg, fps, scale_cm_per_px=0.0):
    seg=seg.copy()
    hipx=(seg.LHip_x.to_numpy(float)+seg.RHip_x.to_numpy(float))/2.0
    shx=(seg.LShoulder_x.to_numpy(float)+seg.RShoulder_x.to_numpy(float))/2.0
    # Proxy del centro de masa: combinación tronco-pelvis, no CoM segmentario 3D.
    seg["com_proxy_x"] = 0.65*hipx + 0.35*shx
    lmask, _, _ = _support_mask_2d(seg, "L", fps); rmask, _, _ = _support_mask_2d(seg, "R", fps)
    seg["L_support_2d"]=lmask; seg["R_support_2d"]=rmask
    lx,_=_foot_centroid(seg,"L"); rx,_=_foot_centroid(seg,"R")
    seg["bos_px"]=np.abs(rx-lx)
    seg["L_dynamic_valgus"]=[_medial_knee_deviation_angle(r,"L") for _,r in seg.iterrows()]
    seg["R_dynamic_valgus"]=[_medial_knee_deviation_angle(r,"R") for _,r in seg.iterrows()]
    if scale_cm_per_px and scale_cm_per_px>0:
        seg["com_proxy_cm"]=(seg.com_proxy_x-np.nanmedian(seg.com_proxy_x))*float(scale_cm_per_px)
        seg["bos_cm"]=seg.bos_px*float(scale_cm_per_px)
    else:
        seg["com_proxy_cm"]=np.nan
        seg["bos_cm"]=np.nan
    return seg

def visibility_pct(seg, names, threshold=0.5):
    cols = [f"{n}_score" for n in names if f"{n}_score" in seg.columns]
    if not cols:
        return np.nan
    return float((seg[cols].min(axis=1) >= threshold).mean() * 100)


def _sync_signal(df):
    """Señal corporal vertical normalizada para estimar desfase temporal entre cámaras."""
    if df is None or df.empty:
        return np.array([]), np.array([])
    hip_y = (df["LHip_y"].to_numpy(float) + df["RHip_y"].to_numpy(float)) / 2.0
    sh_y = (df["LShoulder_y"].to_numpy(float) + df["RShoulder_y"].to_numpy(float)) / 2.0
    torso = np.abs(hip_y - sh_y)
    torso[torso < 1.0] = np.nan
    sig = hip_y / torso
    sig = pd.Series(sig).interpolate(limit_direction="both").rolling(7, center=True, min_periods=1).mean().to_numpy()
    sig = np.gradient(sig)
    return df["frame"].to_numpy(float), sig


def estimate_sync_offset(df1, fps1, df2, fps2, max_offset_s=2.0, resample_hz=50):
    """
    Estima desfase cam02 vs cam01 por correlación de movimiento vertical corporal.
    offset_s > 0: el mismo evento aparece más tarde en cam02; para alinear, avanzar/recortar cam02 ese tiempo.
    Es una heurística experimental, no sustituto de sincronización hardware.
    """
    f1, s1 = _sync_signal(df1); f2, s2 = _sync_signal(df2)
    if len(s1) < 20 or len(s2) < 20 or fps1 <= 0 or fps2 <= 0:
        return 0.0, np.nan, "No calculable"
    t1 = f1 / float(fps1); t2 = f2 / float(fps2)
    dur = min(float(np.nanmax(t1)), float(np.nanmax(t2)))
    if dur < 2.0:
        return 0.0, np.nan, "No calculable"
    grid = np.arange(0.0, dur, 1.0 / resample_hz)
    a = np.interp(grid, t1, s1); b = np.interp(grid, t2, s2)
    a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
    b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-8)
    max_k = int(round(max_offset_s * resample_hz))
    best_k, best_corr = 0, -np.inf
    for k in range(-max_k, max_k + 1):
        if k > 0:
            x, y = a[:-k], b[k:]
        elif k < 0:
            x, y = a[-k:], b[:k]
        else:
            x, y = a, b
        if len(x) < resample_hz:
            continue
        c = np.corrcoef(x, y)[0,1]
        if np.isfinite(c) and c > best_corr:
            best_corr, best_k = float(c), int(k)
    offset_s = best_k / float(resample_hz)
    quality = "Alta" if best_corr >= 0.65 else ("Moderada" if best_corr >= 0.40 else "Baja")
    return float(offset_s), float(best_corr), quality


def prefix_metrics(metrics, prefix, label_prefix):
    out = []
    for m in metrics:
        x = dict(m)
        x["key"] = f"{prefix}_{m['key']}"
        x["label"] = f"{label_prefix} · {m['label']}"
        out.append(x)
    return out


def camera_pose_quality(df):
    if df is None or df.empty:
        return {"tracking": np.nan, "good_frames": np.nan, "foot_visibility": np.nan, "lower_visibility": np.nan}
    score_cols = [f"{n}_score" for n in LOWER_BODY if f"{n}_score" in df.columns]
    tracking = float(df[score_cols].mean(axis=1).mean()) if score_cols else np.nan
    good = float((df[score_cols].min(axis=1) >= 0.5).mean() * 100) if score_cols else np.nan
    foot = visibility_pct(df, FOOT_POINTS, 0.5)
    lower = visibility_pct(df, LOWER_BODY, 0.5)
    return {"tracking": tracking, "good_frames": good, "foot_visibility": foot, "lower_visibility": lower}


def readiness_3d(mode, q1, q2, sync_quality, calibration_name):
    reasons = []
    if not mode.startswith("2 cámaras"):
        reasons.append("Se requieren dos cámaras.")
    if not calibration_name:
        reasons.append("Falta un perfil de calibración 2 cámaras.")
    if sync_quality not in ("Alta", "Moderada"):
        reasons.append("La sincronización automática es insuficiente o no calculable.")
    for idx, q in enumerate((q1, q2), start=1):
        lv = q.get("lower_visibility", np.nan) if q else np.nan
        if not np.isfinite(lv) or lv < 70:
            reasons.append(f"Visibilidad de tren inferior insuficiente en cámara {idx} (<70%).")
    return len(reasons) == 0, reasons



def _edges(mask):
    """Devuelve inicios y finales de periodos True. Índices relativos al segmento."""
    m=np.asarray(mask,dtype=bool)
    if len(m)==0:
        return np.array([],dtype=int), np.array([],dtype=int)
    d=np.diff(m.astype(int), prepend=int(m[0]))
    starts=np.where(d==1)[0].tolist()
    ends=np.where(d==-1)[0].tolist()
    if m[0]: starts=[0]+starts
    if m[-1]: ends=ends+[len(m)]
    return np.asarray(starts,dtype=int), np.asarray(ends,dtype=int)


def _nearest_valid_indices(starts, ends, n):
    out=[]
    for ic in starts:
        offs=ends[ends>ic]
        if len(offs):
            out.append((int(ic), int(min(offs[0], n-1))))
    return out


def _mean_at_indices(arr, idxs):
    a=np.asarray(arr,float)
    vals=[a[int(i)] for i in idxs if 0<=int(i)<len(a) and np.isfinite(a[int(i)])]
    return float(np.nanmean(vals)) if vals else np.nan


def _window_stat(arr, starts, cycles, lo_pct, hi_pct, fn="max"):
    """Estadística en ventana porcentual del ciclo, usando IC consecutivos como ciclo."""
    a=np.asarray(arr,float); vals=[]
    for j in range(len(starts)-1):
        i0,i1=int(starts[j]),int(starts[j+1])
        if i1-i0<5: continue
        lo=i0+int(round((i1-i0)*lo_pct/100.0)); hi=i0+int(round((i1-i0)*hi_pct/100.0))
        lo=max(i0,min(lo,len(a)-1)); hi=max(lo+1,min(hi,len(a)))
        x=a[lo:hi]; x=x[np.isfinite(x)]
        if not len(x): continue
        vals.append(float(np.nanmax(x) if fn=="max" else np.nanmin(x) if fn=="min" else np.nanmean(x)))
    return float(np.nanmean(vals)) if vals else np.nan



def compute_gait_phase_metrics(seg, fps, is_frontal, expected_stride_s=np.nan,
                               support_summary_l=None, support_summary_r=None,
                               temporal_coherence_ok=True):
    """
    v0.9.2: fases del ciclo a partir de estados de apoyo estabilizados.
    IC/TO siguen siendo eventos cinemáticos 2D estimados.
    """
    lmask = np.asarray(seg["L_support_2d"], bool) if "L_support_2d" in seg else _support_mask_2d(seg, "L", fps, expected_stride_s)[0]
    rmask = np.asarray(seg["R_support_2d"], bool) if "R_support_2d" in seg else _support_mask_2d(seg, "R", fps, expected_stride_s)[0]

    L = support_summary_l or _support_cycle_summary(lmask, fps, expected_stride_s)
    R = support_summary_r or _support_cycle_summary(rmask, fps, expected_stride_s)
    reliable = L["quality"] in ("Alta", "Moderada") and R["quality"] in ("Alta", "Moderada")

    if reliable:
        # v0.9.2: el doble apoyo se calcula SOLO en el dominio cubierto
        # simultáneamente por ciclos válidos de ambos lados y por marcha rectilínea.
        cov_l = np.asarray(L.get("coverage_mask", np.ones(len(lmask), dtype=bool)), bool)
        cov_r = np.asarray(R.get("coverage_mask", np.ones(len(rmask), dtype=bool)), bool)
        valid_domain = cov_l & cov_r
        if "straight_walking_valid" in seg:
            valid_domain &= np.asarray(seg["straight_walking_valid"], bool)

        both = lmask & rmask
        none = (~lmask) & (~rmask)

        if valid_domain.any():
            double_pct_raw = float(np.mean(both[valid_domain]) * 100.0)
            double_s_raw = float(np.sum(both[valid_domain]) / float(fps))
            flight_pct = float(np.mean(none[valid_domain]) * 100.0)
        else:
            double_pct_raw = double_s_raw = flight_pct = np.nan

        swing_l = float(np.nanmean(L["swing"])) if len(L["swing"]) else np.nan
        swing_r = float(np.nanmean(R["swing"])) if len(R["swing"]) else np.nan
        swing_asym = abs(swing_l - swing_r) / ((swing_l + swing_r) / 2.0) * 100.0 if np.isfinite(swing_l) and np.isfinite(swing_r) and (swing_l + swing_r) > 0 else np.nan
        stance_pct_l = float(np.nanmean(L["stance_pct"])) if len(L["stance_pct"]) else np.nan
        stance_pct_r = float(np.nanmean(R["stance_pct"])) if len(R["stance_pct"]) else np.nan
        stance_pct = float(np.nanmean([stance_pct_l, stance_pct_r])) if np.isfinite(stance_pct_l) and np.isfinite(stance_pct_r) else np.nan
        swing_pct = 100.0 - stance_pct if np.isfinite(stance_pct) else np.nan

        # Control físico interno: en marcha ordinaria el detector no debería
        # producir una fase aérea apreciable. Si ocurre, no se publica doble apoyo.
        expected_double = max(0.0, stance_pct_l + stance_pct_r - 100.0) if np.isfinite(stance_pct_l) and np.isfinite(stance_pct_r) else np.nan
        ds_discrepancy = abs(double_pct_raw - expected_double) if np.isfinite(double_pct_raw) and np.isfinite(expected_double) else np.nan
        ds_ok = (
            np.isfinite(double_pct_raw)
            and np.isfinite(flight_pct)
            and flight_pct <= 5.0
            and (not np.isfinite(ds_discrepancy) or ds_discrepancy <= 5.0)
            and bool(temporal_coherence_ok)
        )
        if ds_ok:
            double_pct = double_pct_raw
            double_s = double_s_raw
            phase_quality = "Experimental · ciclos rectilíneos validados"
        else:
            double_pct = double_s = np.nan
            phase_quality = "Parcialmente fiable · doble apoyo suprimido por coherencia temporal/física"
    else:
        double_pct = double_s = swing_l = swing_r = swing_asym = stance_pct = swing_pct = np.nan
        flight_pct = expected_double = ds_discrepancy = np.nan
        phase_quality = "No fiable"

    out = [
        {"key":"temporal_segmentation_score","label":"Consistencia de segmentación apoyo/oscilación","value":float(min(L["score"], R["score"])),"unit":"/100","quality":"Alta" if reliable and min(L["score"],R["score"])>=75 else ("Moderada" if reliable else "No fiable"),"notes":f"Control ciclo a ciclo. Izq: {L['reason']}. Der: {R['reason']}. No es un índice clínico."},
        {"key":"stance_pct_2d","label":"Fase de apoyo estimada","value":stance_pct,"unit":"% ciclo","quality":phase_quality,"notes":"IC→TO estimados mediante estados temporales estabilizados; referencia ~60% solo como contexto."},
        {"key":"swing_pct_2d","label":"Fase de oscilación estimada","value":swing_pct,"unit":"% ciclo","quality":phase_quality,"notes":"TO→siguiente IC; se anula si falla la consistencia temporal."},
        {"key":"double_support_pct_2d","label":"Doble apoyo estimado","value":double_pct,"unit":"% dominio válido","quality":phase_quality,"notes":"v0.9.2: intersección directa de apoyos dentro del dominio rectilíneo común. Se suprime si hay fase aérea >5%, discordancia >5 puntos porcentuales respecto a la ocupación de apoyo o fallo de coherencia temporal global."},
        {"key":"double_support_expected_from_stance_pct","label":"Doble apoyo esperado por ocupación de apoyo","value":expected_double,"unit":"% ciclo","quality":"Control interno","notes":"Máx(0, apoyo_I% + apoyo_D% − 100). Se usa como comprobación matemática, no como medida clínica independiente."},
        {"key":"unsupported_flight_pct_2d","label":"Frames sin apoyo detectado","value":flight_pct,"unit":"% dominio válido","quality":"Control interno","notes":"En marcha ordinaria un valor relevante sugiere fallo de segmentación; >5% invalida el doble apoyo."},
        {"key":"double_support_internal_discrepancy_pct","label":"Discrepancia interna de doble apoyo","value":ds_discrepancy,"unit":"puntos %","quality":"Control interno","notes":"Diferencia entre solapamiento observado y esperado por ocupación de apoyo; >10 puntos invalida el doble apoyo."},
        {"key":"double_support_time_2d","label":"Tiempo acumulado de doble apoyo","value":double_s,"unit":"s","quality":phase_quality,"notes":"Tiempo acumulado del segmento; no duración media por ciclo."},
        {"key":"swing_time_l_2d","label":"Tiempo de oscilación izquierdo estimado","value":swing_l,"unit":"s","quality":phase_quality,"notes":"TO izquierdo→siguiente IC izquierdo."},
        {"key":"swing_time_r_2d","label":"Tiempo de oscilación derecho estimado","value":swing_r,"unit":"s","quality":phase_quality,"notes":"TO derecho→siguiente IC derecho."},
        {"key":"swing_asymmetry_2d","label":"Asimetría de oscilación estimada","value":swing_asym,"unit":"%","quality":phase_quality,"notes":"Diferencia relativa D/I solo si la segmentación es consistente."},
        {"key":"initial_contacts_l_n","label":"Contactos iniciales izquierdos estimados","value":float(len(L["ic"])) if reliable else np.nan,"unit":"eventos","quality":phase_quality,"notes":"Eventos cinemáticos 2D estimados; no heel-strikes validados."},
        {"key":"initial_contacts_r_n","label":"Contactos iniciales derechos estimados","value":float(len(R["ic"])) if reliable else np.nan,"unit":"eventos","quality":phase_quality,"notes":"Eventos cinemáticos 2D estimados; no heel-strikes validados."},
    ]

    if not reliable:
        return out

    if is_frontal:
        for side, label, summary in [("L","izquierda",L),("R","derecha",R)]:
            ics = summary["ic"]
            foot = seg[f"{side}_foot_progress_proj"].to_numpy(float)
            rear = seg[f"{side}_rearfoot_tilt_proj"].to_numpy(float)
            valg = seg[f"{side}_dynamic_valgus"].to_numpy(float)
            out += [
                {"key":f"initial_contact_foot_{side.lower()}_deg","label":f"Orientación del pie {label} en contacto inicial estimado","value":_mean_at_indices(foot,ics),"unit":"°","quality":phase_quality,"notes":"Orientación distal 2D en IC estimado; no confirma talón/antepié."},
                {"key":f"initial_contact_rearfoot_{side.lower()}_deg","label":f"Retropié {label} en contacto inicial estimado","value":_mean_at_indices(rear,ics),"unit":"°","quality":phase_quality,"notes":"Inclinación proyectada del retropié en IC estimado."},
                {"key":f"loading_knee_{side.lower()}_deg","label":f"Desviación medial rodilla {label} en respuesta a la carga","value":_window_stat(valg,ics,None,0,10,"max"),"unit":"°","quality":phase_quality,"notes":"Máximo 2D en ventana 0–10% de ciclos aceptados; el valgo real es multiplanar."},
                {"key":f"terminal_foot_{side.lower()}_deg","label":f"Orientación distal pie {label} en pre-oscilación","value":_window_stat(foot,ics,None,50,60,"mean"),"unit":"°","quality":phase_quality,"notes":"Ventana 50–60% solo en ciclos aceptados."},
            ]
    else:
        for side, label, summary in [("L","izquierda",L),("R","derecha",R)]:
            ics = summary["ic"]
            ankle = seg[f"{side}_ankle_angle"].to_numpy(float)
            knee = seg[f"{side}_knee_flex"].to_numpy(float)
            out += [
                {"key":f"initial_contact_foot_{side.lower()}_deg","label":f"Ángulo tobillo-pie {label} en contacto inicial estimado","value":_mean_at_indices(ankle,ics),"unit":"°","quality":phase_quality,"notes":"Ángulo sagital 2D en IC estimado; no confirma heel-strike mediante fuerza."},
                {"key":f"loading_knee_{side.lower()}_deg","label":f"Flexión rodilla {label} en respuesta a la carga","value":_window_stat(knee,ics,None,0,10,"max"),"unit":"°","quality":phase_quality,"notes":"Máxima flexión proyectada en ventana 0–10% de ciclos aceptados."},
                {"key":f"terminal_foot_{side.lower()}_deg","label":f"Ángulo tobillo-pie {label} en pre-oscilación","value":_window_stat(ankle,ics,None,50,60,"mean"),"unit":"°","quality":phase_quality,"notes":"Ventana 50–60% de ciclos aceptados."},
            ]
    return out


def compute_metrics(df, fps, start_frame, end_frame, view, assistive_device="Sin ayuda", scale_cm_per_px=0.0):
    seg = df[(df.frame >= start_frame) & (df.frame <= end_frame)].copy()
    if len(seg) < max(30, int(fps * 2)):
        raise ValueError("El segmento seleccionado es demasiado corto.")
    seg = add_angle_columns(seg)
    is_frontal = "Frontal" in (view or "")
    if is_frontal:
        seg = add_frontal_columns(seg)
        seg = add_frontal_advanced(seg, fps, scale_cm_per_px)

    score_cols = [f"{n}_score" for n in LOWER_BODY]
    mean_tracking = float(seg[score_cols].mean(axis=1).mean())
    good_frames = float((seg[score_cols].min(axis=1) >= 0.5).mean() * 100)
    foot_visible = visibility_pct(seg, FOOT_POINTS, 0.5)
    upper_visible = visibility_pct(seg, UPPER_BODY, 0.5)
    q = quality_label(mean_tracking)
    assisted = assistive_device != "Sin ayuda"

    # v0.8.1 · dominio rectilíneo + ciclos IC→IC coherentes.
    # El giro y las transiciones no deben contaminar CV, asimetría ni doble apoyo.
    straight_mask, straight_info = _automatic_straight_walking_mask(seg, fps)
    seg["straight_walking_valid"] = straight_mask

    # Proxy distal se conserva solo para una primera estimación amplia del periodo.
    ly = (seg.LAnkle_y.to_numpy() + seg.LHeel_y.to_numpy() + seg.LBigToe_y.to_numpy()) / 3.0
    ry = (seg.RAnkle_y.to_numpy() + seg.RHeel_y.to_numpy() + seg.RBigToe_y.to_numpy()) / 3.0
    diff = rolling_smooth(ry - ly, 7)
    crossings = zero_crossings(diff)
    if len(crossings) > 1:
        kept = [crossings[0]]
        min_gap = max(1, int(round(0.25 * fps)))
        for c in crossings[1:]:
            if c - kept[-1] >= min_gap:
                kept.append(c)
        crossings = np.asarray(kept, dtype=int)

    # Periodo inicial robusto para estabilizar el detector de apoyo.
    intervals0 = np.diff(crossings) / fps if len(crossings) >= 2 else np.asarray([])
    mean_alt0 = float(np.nanmedian(intervals0)) if len(intervals0) >= 3 else np.nan
    expected_stride_s = (2.0 * mean_alt0) if np.isfinite(mean_alt0) and mean_alt0 > 0 else np.nan

    l_support, _, _ = _support_mask_2d(seg, "L", fps, expected_stride_s)
    r_support, _, _ = _support_mask_2d(seg, "R", fps, expected_stride_s)
    seg["L_support_2d"] = l_support
    seg["R_support_2d"] = r_support

    l_cycle0 = _support_cycle_summary(l_support, fps, expected_stride_s)
    r_cycle0 = _support_cycle_summary(r_support, fps, expected_stride_s)

    # Excluir ciclos que invaden giro/transiciones.
    l_cycle_orig = _filter_support_summary_to_mask(l_cycle0, straight_mask, fps, min_fraction=0.90)
    r_cycle_orig = _filter_support_summary_to_mask(r_cycle0, straight_mask, fps, min_fraction=0.90)

    # v0.10.9 · fallback robusto si la cadena original no supera QC temporal amplio.
    l_sel,r_sel,support_event_source,support_event_info = _select_temporal_event_chain(
        seg.reset_index(drop=True), fps, l_cycle_orig, r_cycle_orig
    )
    if support_event_source == "robust_distal":
        l_cycle = _filter_support_summary_to_mask(l_sel, straight_mask, fps, min_fraction=0.90)
        r_cycle = _filter_support_summary_to_mask(r_sel, straight_mask, fps, min_fraction=0.90)
    else:
        l_cycle,r_cycle = l_cycle_orig,r_cycle_orig

    support_reliable = l_cycle["quality"] in ("Alta", "Moderada") and r_cycle["quality"] in ("Alta", "Moderada")

    # v0.9.2 · Separación explícita de funciones:
    # - CV: robusto por lado (v0.9.1, preservado).
    # - Cadencia: secuencia anatómica L-R-L-R canónica.
    # - Cadencia ipsilateral: control secundario SIN filtro de outliers del CV.
    # - Cierre físico: apoyo L + apoyo R + doble apoyo observado.
    cycle_timing = _cycle_timing_metrics(l_cycle, r_cycle, fps) if support_reliable else {
        "cadence": np.nan, "mean_stride": np.nan, "cv": np.nan,
        "cv_left": np.nan, "cv_right": np.nan,
        "l_mean_stride": np.nan, "r_mean_stride": np.nan,
        "n_left": 0, "n_right": 0, "n_cycles": 0,
        "outliers_left": 0, "outliers_right": 0,
        "quality": "No fiable",
        "reason": "segmentación apoyo/oscilación insuficiente"
    }

    canonical = _canonical_gait_timeline(l_cycle, r_cycle, fps) if support_reliable else {
        "cadence": np.nan, "mean_step": np.nan, "asym": np.nan,
        "events": [], "n_intervals": 0, "quality": "No fiable",
        "reason": "segmentación apoyo/oscilación insuficiente"
    }

    cadence_steps = canonical.get("cadence", np.nan)
    cadence_stride, mean_stride_raw, n_stride_raw = _raw_stride_cadence(l_cycle, r_cycle) if support_reliable else (np.nan, np.nan, 0)

    # v0.10.6 · detector de ritmo completamente independiente del contacto.
    # Esto permite estimar cadencia/asimetría cuando el tracking articular es
    # excelente pero IC/TO o las máscaras de apoyo no cierran correctamente.
    kin_alt = _kinematic_alternation_metrics(seg, fps)
    cadence_kin = kin_alt.get("cadence", np.nan)
    asym_kin = kin_alt.get("asym", np.nan)
    cv_kin = kin_alt.get("cv", np.nan)

    closure = _temporal_closure_from_masks(
        l_support, r_support, l_cycle, r_cycle, straight_mask, fps
    ) if support_reliable else {
        "double_raw": np.nan, "flight_pct": np.nan,
        "stance_pct_l": np.nan, "stance_pct_r": np.nan,
        "double_expected": np.nan, "double_discrepancy": np.nan,
        "closure_stride": np.nan, "closure_cadence": np.nan,
    }
    cadence_closure = closure.get("closure_cadence", np.nan)

    # CV permanece exactamente en la lógica robusta v0.9.1.
    cv_alt = cycle_timing.get("cv", np.nan)

    # Concordancias entre tres lecturas del MISMO fenómeno temporal.
    err_step_stride = _pct_disagreement(cadence_steps, cadence_stride)
    err_step_closure = _pct_disagreement(cadence_steps, cadence_closure)
    err_stride_closure = _pct_disagreement(cadence_stride, cadence_closure)

    available_cadences = [x for x in (cadence_steps, cadence_stride, cadence_closure) if np.isfinite(x)]
    pair_errors = [x for x in (err_step_stride, err_step_closure, err_stride_closure) if np.isfinite(x)]
    max_pair_error = float(max(pair_errors)) if pair_errors else np.nan

    # v0.10.6 · Jerarquía de cadencia:
    # A) pasos anatómicos IC L/R;
    # B) alternancia cinemática de pies (independiente de apoyo);
    # C) ciclos IC→IC ipsilaterales.
    #
    # El cierre apoyo/doble apoyo es QC, no requisito para contar pasos/min.
    n_step_intervals = int(canonical.get("n_intervals", 0) or 0)
    cadence_step_ok = np.isfinite(cadence_steps) and n_step_intervals >= 3
    cadence_kin_ok = np.isfinite(cadence_kin) and int(kin_alt.get("n_intervals",0)) >= 4
    cadence_stride_ok = np.isfinite(cadence_stride) and int(n_stride_raw) >= 2

    candidates_primary = []
    if cadence_step_ok:
        candidates_primary.append(("IC bilateral", float(cadence_steps)))
    if cadence_kin_ok:
        candidates_primary.append(("alternancia cinemática", float(cadence_kin)))
    if cadence_stride_ok:
        candidates_primary.append(("IC ipsilateral", float(cadence_stride)))

    cadence = np.nan
    cadence_quality = "No calculable"
    cadence_reason = "sin periodicidad distal suficiente"

    if cadence_step_ok and cadence_kin_ok:
        err = _pct_disagreement(cadence_steps, cadence_kin)
        if np.isfinite(err) and err <= 12.0:
            cadence = float(np.nanmean([cadence_steps, cadence_kin]))
            cadence_quality = "Alta" if err <= 7.0 else "Moderada"
            cadence_reason = f"IC bilateral y alternancia cinemática concuerdan ({err:.1f}%)"
        else:
            # Cuando las máscaras IC fallan pero el movimiento distal tiene varios
            # ciclos repetidos, preferir el detector cinemático y degradar calidad.
            cadence = float(cadence_kin)
            cadence_quality = "Moderada · detector cinemático"
            cadence_reason = (
                f"alternancia distal publicada; IC bilateral discrepa {err:.1f}%"
                if np.isfinite(err) else "alternancia distal con IC bilateral no comparable"
            )
    elif cadence_kin_ok:
        cadence = float(cadence_kin)
        cadence_quality = kin_alt.get("quality","Moderada") + " · detector cinemático"
        cadence_reason = kin_alt.get("reason","alternancia distal")
    elif cadence_step_ok:
        cadence = float(cadence_steps)
        cadence_quality = canonical.get("quality","Moderada")
        cadence_reason = canonical.get("reason","cadena anatómica IC")
    elif cadence_stride_ok:
        cadence = float(cadence_stride)
        cadence_quality = "Baja · ciclos ipsilaterales"
        cadence_reason = f"{n_stride_raw} ciclos IC→IC válidos"

    mean_alt = (60.0/cadence) if np.isfinite(cadence) and cadence > 0 else np.nan
    mean_stride_cycle = (120.0/cadence) if np.isfinite(cadence) and cadence > 0 else np.nan

    # Coherencia global sigue siendo estricta para métricas de CONTACTO.
    if len(available_cadences) >= 3:
        temporal_coherence_ok = bool(max_pair_error <= 10.0)
        temporal_coherence_quality = "Alta" if max_pair_error <= 6.0 else ("Moderada" if temporal_coherence_ok else "No fiable")
    elif len(available_cadences) == 2:
        temporal_coherence_ok = bool(max_pair_error <= 10.0)
        temporal_coherence_quality = "Moderada" if temporal_coherence_ok else "No fiable"
    else:
        temporal_coherence_ok = False
        temporal_coherence_quality = "No fiable"

    # Asimetría temporal: IC anatómico si el cierre es bueno; si no, usar
    # alternancia cinemática global, claramente etiquetada como tal.
    if temporal_coherence_ok and np.isfinite(canonical.get("asym", np.nan)):
        asym = canonical.get("asym", np.nan)
        asym_quality = temporal_coherence_quality
        asym_source = "IC anatómicos L→R/R→L"
    elif np.isfinite(asym_kin) and int(kin_alt.get("n_intervals",0)) >= 4:
        asym = float(asym_kin)
        asym_quality = kin_alt.get("quality","Baja") + " · cinemática"
        asym_source = "alternancia distal P→T/T→P"
    else:
        asym = np.nan
        asym_quality = "No calculable"
        asym_source = "sin eventos temporales suficientes"

    # CV: conservar el de ciclos por lado cuando existe; si no, usar el CV del
    # intervalo de alternancia distal para no dejar el registro vacío.
    if not np.isfinite(cv_alt) and np.isfinite(cv_kin):
        cv_alt = float(cv_kin)
        cv_quality_override = kin_alt.get("quality","Baja") + " · cinemática"
    else:
        cv_quality_override = None

    if support_reliable:
        stance_l = float(np.nanmean(l_cycle["stance"])) if len(l_cycle["stance"]) else np.nan
        stance_r = float(np.nanmean(r_cycle["stance"])) if len(r_cycle["stance"]) else np.nan
    else:
        stance_l = stance_r = np.nan

    stance_asym = abs(stance_l-stance_r)/((stance_l+stance_r)/2.0)*100.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and (stance_l+stance_r)>0 else np.nan
    stance_longer = 1.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and stance_l>stance_r else (2.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and stance_r>stance_l else 0.0)

    segment_duration_s = float((seg.frame.max() - seg.frame.min() + 1) / fps)
    valid_duration_s = float(np.sum(straight_mask) / fps)

    detected_steps = int(len(canonical.get("events", [])))
    cadence_count_segment = (60.0 * detected_steps / valid_duration_s) if valid_duration_s > 0 and detected_steps > 0 else np.nan
    expected_steps = (cadence * valid_duration_s / 60.0) if np.isfinite(cadence) else np.nan
    consistency_error = (
        abs(detected_steps - expected_steps) / max(expected_steps, 1.0) * 100.0
        if np.isfinite(expected_steps) else np.nan
    )

    if not np.isfinite(cadence):
        consistency_quality = "Revisar"
    elif detected_steps < 4 or not np.isfinite(consistency_error):
        consistency_quality = "Limitada"
    elif consistency_error <= 10:
        consistency_quality = "Alta"
    elif consistency_error <= 20:
        consistency_quality = "Moderada"
    else:
        consistency_quality = "Revisar"

    aid_note = f" Marcha con ayuda técnica: {assistive_device}; revisar oclusiones." if assisted else ""
    metrics = [
        {"key":"support_event_source","label":"Fuente de eventos temporales","value":support_event_source,"unit":"","quality":support_event_info.get("quality","") if isinstance(support_event_info,dict) else "","notes":support_event_info.get("reason","") if isinstance(support_event_info,dict) else ""},
        {"key":"support_event_quality","label":"Calidad cadena IC/TO alternativa","value":support_event_info.get("quality","") if isinstance(support_event_info,dict) else "","unit":"","quality":support_event_info.get("quality","") if isinstance(support_event_info,dict) else "","notes":"QC interno; no equivale a validación con plataforma de fuerza."},
        {"key":"tracking_mean","label":"Confianza media del tracking","value":mean_tracking,"unit":"","quality":q,"notes":"Media HALPE26 del tren inferior."+aid_note},
        {"key":"good_frames_pct","label":"Frames con tren inferior visible ≥0,50","value":good_frames,"unit":"%","quality":q,"notes":"Puntos principales del tren inferior ≥0,50."+aid_note},
        {"key":"foot_visibility_pct","label":"Visibilidad de pie/tobillo","value":foot_visible,"unit":"%","quality":quality_label((foot_visible or 0)/100),"notes":"Tobillo, talón y antepié. Fundamental para métricas del pie."+aid_note},
        {"key":"upper_visibility_pct","label":"Visibilidad del tren superior","value":upper_visible,"unit":"%","quality":quality_label((upper_visible or 0)/100),"notes":"Hombros, codos y muñecas; puede disminuir con muletas/caminador."+aid_note},
        {"key":"straight_walking_usable_pct","label":"Tramo rectilíneo utilizable","value":float(straight_info["usable_pct"]),"unit":"% segmento","quality":"Control interno","notes":straight_info["reason"]+"; v0.9.2 excluye giro/transiciones de las métricas temporales sensibles."},
        {"key":"turn_transition_excluded_pct","label":"Giro/transiciones excluidos","value":float(straight_info["excluded_pct"]),"unit":"% segmento","quality":"Control interno","notes":"Exclusión automática por orientación corporal/inversión de trayectoria; no es una métrica clínica."},
        {"key":"cadence_exp","label":"Cadencia estimada","value":cadence,"unit":"pasos/min","quality":cadence_quality,"notes":"v0.10.6: jerarquía IC bilateral + alternancia cinemática distal + ciclos IC→IC. El cierre de apoyo no impide contar pasos/min si la periodicidad articular es suficiente. "+cadence_reason},
        {"key":"cadence_kinematic","label":"Cadencia por alternancia cinemática","value":cadence_kin,"unit":"pasos/min","quality":kin_alt.get("quality","No fiable"),"notes":"Estimación independiente de las máscaras de apoyo a partir de máximos/mínimos alternantes de la trayectoria relativa de ambos pies."},
        {"key":"kinematic_alternation_events","label":"Eventos de alternancia cinemática","value":float(len(kin_alt.get("event_frames",[]))),"unit":"eventos","quality":kin_alt.get("quality","No fiable"),"notes":"Extremos alternantes del movimiento distal usados como detector de periodicidad; no equivalen a heel-strikes de plataforma."},
        {"key":"kinematic_periodicity","label":"Periodicidad cinemática","value":kin_alt.get("periodicity",np.nan),"unit":"r autocorr","quality":kin_alt.get("quality","No fiable"),"notes":"Pico de autocorrelación de la señal distal detrendida; control interno de repetición."},
        {"key":"step_events_detected","label":"Contactos IC anatómicos en cadena válida","value":float(detected_steps),"unit":"eventos","quality":canonical.get("quality","No fiable"),"notes":"Contactos iniciales L/R derivados de ciclos de apoyo y ordenados en una única secuencia alternante; no son heel-strikes de plataforma de fuerzas."},
        {"key":"segment_duration_s","label":"Duración del segmento analizado","value":segment_duration_s,"unit":"s","quality":"Directa","notes":"Duración temporal total del intervalo seleccionado."},
        {"key":"valid_straight_duration_s","label":"Duración rectilínea utilizable","value":valid_duration_s,"unit":"s","quality":"Control interno","notes":"Tiempo realmente usado tras excluir bordes/giro/transiciones."},
        {"key":"stride_cycle_duration_s","label":"Duración de zancada desde cadena alternante","value":mean_stride_cycle,"unit":"s","quality":temporal_coherence_quality if np.isfinite(mean_stride_cycle) else "No fiable","notes":"Dos intervalos de paso de la línea temporal anatómica v0.9.2; se suprime si falla la coherencia temporal."},
        {"key":"cadence_count_segment","label":"Cadencia por recuento/duración rectilínea","value":cadence_count_segment,"unit":"pasos/min","quality":"Control interno" if np.isfinite(cadence_count_segment) else "No calculable","notes":"60 × contactos aceptados / duración rectilínea utilizable; solo control secundario."},
        {"key":"expected_steps_from_cadence","label":"Eventos esperados por cadencia × duración válida","value":expected_steps,"unit":"eventos","quality":"Control interno" if np.isfinite(expected_steps) else "No calculable","notes":"Comprobación interna sobre el dominio rectilíneo: cadencia × duración válida / 60."},
        {"key":"cadence_contact_crosscheck_error_pct","label":"Discrepancia cadencia pasos vs ciclos ipsilaterales","value":err_step_stride,"unit":"%","quality":"Alta" if np.isfinite(err_step_stride) and err_step_stride<=10 else ("Revisar" if np.isfinite(err_step_stride) else "No calculable"),"notes":"Control v0.9.2 entre la cadena L-R anatómica y los ciclos IC→IC ipsilaterales sin filtrado del CV."},
        {"key":"cadence_closure_crosscheck_error_pct","label":"Discrepancia cadencia pasos vs cierre apoyo/doble apoyo","value":err_step_closure,"unit":"%","quality":"Alta" if np.isfinite(err_step_closure) and err_step_closure<=10 else ("Revisar" if np.isfinite(err_step_closure) else "No calculable"),"notes":"Contrasta la cadencia de pasos con la cadencia físicamente compatible con apoyo izquierdo + apoyo derecho + doble apoyo observado."},
        {"key":"cadence_stride_closure_error_pct","label":"Discrepancia ciclos vs cierre físico","value":err_stride_closure,"unit":"%","quality":"Alta" if np.isfinite(err_stride_closure) and err_stride_closure<=10 else ("Revisar" if np.isfinite(err_stride_closure) else "No calculable"),"notes":"Tercer control independiente de coherencia temporal."},
        {"key":"temporal_coherence_max_error_pct","label":"Discordancia temporal máxima","value":max_pair_error,"unit":"%","quality":temporal_coherence_quality,"notes":"Máxima discrepancia entre cadencia por pasos anatómicos, ciclos ipsilaterales y cierre apoyo/doble apoyo. Umbral operativo de publicación: ≤10%."},
        {"key":"temporal_coherence_flag","label":"Coherencia temporal global","value":1.0 if temporal_coherence_ok else 0.0,"unit":"bool","quality":temporal_coherence_quality,"notes":"1 = cadencia/asimetría/doble apoyo superan el control de coherencia v0.9.2; 0 = se suprimen las métricas incompatibles."},
        {"key":"cadence_candidate_steps","label":"Cadencia candidata por pasos anatómicos","value":cadence_steps,"unit":"pasos/min","quality":"Control interno","notes":"No se interpreta aisladamente; candidato primario antes del cierre de coherencia."},
        {"key":"cadence_candidate_stride","label":"Cadencia candidata por ciclos ipsilaterales","value":cadence_stride,"unit":"pasos/min","quality":"Control interno","notes":"120 / duración media de ciclos IC→IC plausibles, sin aplicar el filtro estadístico usado para CV."},
        {"key":"cadence_candidate_closure","label":"Cadencia candidata por cierre apoyo/doble apoyo","value":cadence_closure,"unit":"pasos/min","quality":"Control interno","notes":"120 / [(apoyo_L + apoyo_R)/(1 + doble_apoyo_fracción)], solo si no existe fase aérea relevante."},
        {"key":"step_count_consistency_error_pct","label":"Discrepancia recuento-cadencia-duración","value":consistency_error,"unit":"%","quality":consistency_quality,"notes":"Diferencia relativa entre eventos detectados y eventos esperados por cadencia × duración. Sirve como control de consistencia, no como validación clínica."},
        {"key":"alternation_interval","label":"Intervalo medio de paso anatómico","value":mean_alt,"unit":"s","quality":temporal_coherence_quality if np.isfinite(mean_alt) else "No calculable","notes":"IC-L→IC-R o IC-R→IC-L de la cadena anatómica v0.9.2."},
        {"key":"regularity_cv","label":"Variabilidad temporal global robusta (CV)","value":cv_alt,"unit":"%","quality":(cv_quality_override if cv_quality_override else cycle_timing.get("quality","No fiable")) if np.isfinite(cv_alt) else "No calculable","notes":"v0.10.6: usa CV por ciclos IC→IC cuando es válido; si esos ciclos fallan pero la alternancia distal es repetitiva, informa el CV de intervalos cinemáticos con calidad explícita."},
        {"key":"regularity_cv_left","label":"CV temporal izquierdo","value":cycle_timing.get("cv_left"),"unit":"%","quality":cycle_timing.get("quality","No fiable"),"notes":"CV de duración IC→IC de la extremidad izquierda tras rechazo robusto de ciclos atípicos."},
        {"key":"regularity_cv_right","label":"CV temporal derecho","value":cycle_timing.get("cv_right"),"unit":"%","quality":cycle_timing.get("quality","No fiable"),"notes":"CV de duración IC→IC de la extremidad derecha tras rechazo robusto de ciclos atípicos."},
        {"key":"valid_cycles_left","label":"Ciclos válidos izquierdos","value":cycle_timing.get("n_left"),"unit":"ciclos","quality":"Control interno","notes":"Número de ciclos IC→IC izquierdos usados para cadencia/CV."},
        {"key":"valid_cycles_right","label":"Ciclos válidos derechos","value":cycle_timing.get("n_right"),"unit":"ciclos","quality":"Control interno","notes":"Número de ciclos IC→IC derechos usados para cadencia/CV."},
        {"key":"cycle_outliers_left","label":"Ciclos atípicos excluidos izquierdos","value":cycle_timing.get("outliers_left"),"unit":"ciclos","quality":"Control interno","notes":"Ciclos descartados por filtro robusto MAD/mediana."},
        {"key":"cycle_outliers_right","label":"Ciclos atípicos excluidos derechos","value":cycle_timing.get("outliers_right"),"unit":"ciclos","quality":"Control interno","notes":"Ciclos descartados por filtro robusto MAD/mediana."},
        {"key":"temporal_asymmetry_exp","label":"Asimetría temporal global","value":asym,"unit":"%","quality":asym_quality,"notes":"v0.10.6: fuente="+asym_source+". Si los IC anatómicos no son fiables, puede utilizar alternancia distal cinemática; esta describe asimetría temporal de alternancia y no contacto de fuerzas."},
        {"key":"support_segmentation_score_l","label":"Consistencia segmentación apoyo izquierdo","value":float(l_cycle["score"]),"unit":"/100","quality":l_cycle["quality"],"notes":l_cycle["reason"]+"; control matemático, no índice clínico."},
        {"key":"support_segmentation_score_r","label":"Consistencia segmentación apoyo derecho","value":float(r_cycle["score"]),"unit":"/100","quality":r_cycle["quality"],"notes":r_cycle["reason"]+"; control matemático, no índice clínico."},
        {"key":"stance_time_l_2d","label":"Tiempo de apoyo izquierdo estimado 2D","value":stance_l,"unit":"s","quality":"Experimental · ciclo continuo" if np.isfinite(stance_l) else "No fiable","notes":"IC→TO dentro de ciclos continuos estabilizados; se anula si la segmentación no supera el control temporal."},
        {"key":"stance_time_r_2d","label":"Tiempo de apoyo derecho estimado 2D","value":stance_r,"unit":"s","quality":"Experimental · ciclo continuo" if np.isfinite(stance_r) else "No fiable","notes":"IC→TO dentro de ciclos continuos estabilizados; se anula si la segmentación no supera el control temporal."},
        {"key":"stance_asymmetry_2d","label":"Asimetría del tiempo de apoyo estimado 2D","value":stance_asym,"unit":"%","quality":"Experimental · ciclo continuo" if np.isfinite(stance_asym) else "No fiable","notes":"Solo se calcula si ambos lados superan el control de consistencia de ciclo."},
        {"key":"stance_longer_side_code","label":"Extremidad con mayor apoyo estimado (código)","value":stance_longer,"unit":"1=I,2=D","quality":"Experimental" if stance_longer else "No fiable","notes":"1=izquierda; 2=derecha. No equivale a medición con plataforma/plantilla instrumentada."},
    ]
    metrics += compute_gait_phase_metrics(
        seg, fps, is_frontal,
        expected_stride_s=expected_stride_s,
        support_summary_l=l_cycle,
        support_summary_r=r_cycle,
        temporal_coherence_ok=temporal_coherence_ok,
    )

    if not is_frontal:
        for key_base, label_base, col_base in [
            ("knee_flex", "Flexión rodilla", "knee_flex"),
            ("hip_flex", "Flexión cadera", "hip_flex"),
            ("ankle_angle", "Ángulo tobillo-pie", "ankle_angle"),
            ("shoulder_elev", "Elevación hombro", "shoulder_elev"),
        ]:
            vals = {}
            for side, side_name in [("L","izquierda"),("R","derecha")]:
                arr = seg[f"{side}_{col_base}"].to_numpy(dtype=float)
                p95 = float(np.nanpercentile(arr, 95)); rom = robust_rom(arr); vals[side] = p95
                quality = "Condicionada por ayuda técnica" if assisted and key_base=="shoulder_elev" else q
                metrics += [
                    {"key":f"{key_base}_{side.lower()}_p95","label":f"{label_base} {side_name} 2D (P95)","value":p95,"unit":"°","quality":quality,"notes":f"Ángulo 2D proyectado en vista {view}."+aid_note},
                    {"key":f"{key_base}_{side.lower()}_rom","label":f"ROM {label_base.lower()} {side_name} 2D","value":rom,"unit":"°","quality":quality,"notes":"ROM robusto P95-P5; 2D proyectado."+aid_note},
                ]
            metrics.append({"key":f"{key_base}_diff_p95","label":f"Diferencia D/I {label_base.lower()} 2D","value":abs(vals["L"]-vals["R"]),"unit":"°","quality":q,"notes":"Diferencia absoluta P95 D/I; 2D proyectado."})
    else:
        vals = {}
        for side, side_name in [("L","izquierda"),("R","derecha")]:
            kd = seg[f"{side}_frontal_knee_dev"].to_numpy(float)
            fp = seg[f"{side}_foot_progress_proj"].to_numpy(float)
            rf = seg[f"{side}_rearfoot_tilt_proj"].to_numpy(float)
            vals[side] = {
                "knee": float(np.nanpercentile(kd,95)),
                "foot": float(np.nanmedian(fp)),
                "rear": float(np.nanpercentile(np.abs(rf),95)),
            }
            foot_q = q if foot_visible >= 80 else "Baja/condicionada"
            metrics += [
                {"key":f"frontal_knee_dev_{side.lower()}_p95","label":f"Desviación frontal rodilla {side_name} (P95)","value":vals[side]["knee"],"unit":"°","quality":q,"notes":"Magnitud proyectada del eje cadera-rodilla-tobillo. No diagnostica valgo/varo 3D."},
                {"key":f"foot_progress_{side.lower()}_median","label":f"Orientación del pie {side_name} proyectada (mediana)","value":vals[side]["foot"],"unit":"°","quality":foot_q,"notes":"Proxy distal de orientación/rotación en la imagen. No equivale a rotación axial de cadera."},
                {"key":f"rearfoot_tilt_{side.lower()}_p95","label":f"Inclinación retropié {side_name} proyectada (P95 abs.)","value":vals[side]["rear"],"unit":"°","quality":foot_q,"notes":"Eje tobillo-talón proyectado. Puede sugerir cambios de eversión/inversión, pero no mide pronación 3D."},
            ]
        metrics += [
            {"key":"frontal_knee_dev_diff","label":"Diferencia D/I desviación frontal de rodilla","value":abs(vals['L']['knee']-vals['R']['knee']),"unit":"°","quality":q,"notes":"Comparación 2D proyectada."},
            {"key":"foot_progress_diff","label":"Diferencia D/I orientación del pie proyectada","value":abs(vals['L']['foot']-vals['R']['foot']),"unit":"°","quality":q,"notes":"Proxy distal; no atribuir directamente a rotación de cadera."},
            {"key":"rearfoot_tilt_diff","label":"Diferencia D/I inclinación del retropié proyectada","value":abs(vals['L']['rear']-vals['R']['rear']),"unit":"°","quality":q,"notes":"No equivale a pronación clínica."},
            {"key":"pelvis_obliquity_rom","label":"ROM oblicuidad pélvica proyectada","value":robust_rom(seg.pelvis_obliquity),"unit":"°","quality":q,"notes":"P95-P5 de la línea inter-caderas en el plano de imagen."},
            {"key":"shoulder_obliquity_rom","label":"ROM oblicuidad de hombros proyectada","value":robust_rom(seg.shoulder_obliquity),"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"P95-P5 de la línea biacromial en el plano de imagen."+aid_note},
            {"key":"trunk_lateral_lean_rom","label":"ROM inclinación lateral del tronco proyectada","value":robust_rom(seg.trunk_lateral_lean),"unit":"°","quality":q,"notes":"P95-P5 del eje centro de pelvis-centro de hombros respecto a la vertical de la imagen."},
            {"key":"shoulder_pelvis_rel_rom","label":"ROM relación hombros-pelvis proyectada","value":robust_rom(seg.shoulder_pelvis_rel),"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"Variación de la diferencia entre oblicuidad de hombros y pelvis; descriptor 2D del acoplamiento tronco-pelvis."+aid_note},
            {"key":"base_width_relative_median","label":"Anchura de base relativa (tobillos/pelvis)","value":float(np.nanmedian(seg.base_width_relative)),"unit":"ratio","quality":q,"notes":"Anchura proyectada normalizada por anchura pélvica; no es distancia métrica sin calibración."},
        ]
        # Parámetros frontales avanzados, condicionados por tracking y escala espacial.
        com_amp_px = robust_rom(seg.com_proxy_x)
        com_cm = robust_rom(seg.com_proxy_cm) if np.isfinite(seg.com_proxy_cm.to_numpy(float)).any() else np.nan
        if support_reliable:
            ds = seg[seg.L_support_2d & seg.R_support_2d]
            bos_cm = float(np.nanmedian(ds.bos_cm)) if len(ds)>=3 and np.isfinite(ds.bos_cm.to_numpy(float)).any() else np.nan
            l_single = seg[seg.L_support_2d & ~seg.R_support_2d]
            r_single = seg[seg.R_support_2d & ~seg.L_support_2d]
            # Convención de pelvis: + = lado izquierdo elevado / derecho descendido; - = lado izquierdo descendido / derecho elevado.
            trend_r_swing = float(max(0.0, np.nanpercentile(l_single.pelvis_obliquity,95))) if len(l_single)>=3 else np.nan
            trend_l_swing = float(max(0.0, -np.nanpercentile(r_single.pelvis_obliquity,5))) if len(r_single)>=3 else np.nan
        else:
            bos_cm = np.nan
            trend_r_swing = trend_l_swing = np.nan
        valg_l = float(np.nanpercentile(seg.L_dynamic_valgus,95)); valg_r = float(np.nanpercentile(seg.R_dynamic_valgus,95))
        # Acoplamiento tronco-pelvis: correlación y desfase por correlación cruzada.
        a = pd.Series(seg.trunk_lateral_lean).interpolate(limit_direction="both").to_numpy(float)
        b = pd.Series(seg.pelvis_obliquity).interpolate(limit_direction="both").to_numpy(float)
        coupling_r = float(np.corrcoef(a,b)[0,1]) if len(a)>10 and np.nanstd(a)>1e-6 and np.nanstd(b)>1e-6 else np.nan
        phase_deg=np.nan
        if len(a)>20 and np.isfinite(mean_alt) and mean_alt>0:
            aa=(a-np.nanmean(a))/(np.nanstd(a)+1e-8); bb=(b-np.nanmean(b))/(np.nanstd(b)+1e-8)
            maxlag=min(int(round(fps)),len(a)//3); bestlag=0; best=-np.inf
            for lag in range(-maxlag,maxlag+1):
                x,y=(aa[:-lag],bb[lag:]) if lag>0 else ((aa[-lag:],bb[:lag]) if lag<0 else (aa,bb))
                if len(x)>10:
                    c=np.corrcoef(x,y)[0,1]
                    if np.isfinite(c) and abs(c)>best: best=abs(c); bestlag=lag
            stride_period = mean_stride_cycle if np.isfinite(mean_stride_cycle) and mean_stride_cycle > 0 else (2.0 * mean_alt)
            phase_deg = float(bestlag / fps / stride_period * 360.0)
            # v0.9.2: fase circular canónica en [-180, +180).
            phase_deg = float(((phase_deg + 180.0) % 360.0) - 180.0)
        metrics += [
            {"key":"com_lateral_excursion_cm","label":"Oscilación lateral CoM proxy 2D","value":com_cm,"unit":"cm","quality":q if np.isfinite(com_cm) else "Requiere escala","notes":"Proxy tronco-pelvis. Se expresa en cm solo con escala espacial 2D calibrada; no es CoM segmentario 3D."},
            {"key":"com_lateral_excursion_px","label":"Oscilación lateral CoM proxy 2D (imagen)","value":com_amp_px,"unit":"px","quality":q,"notes":"Amplitud robusta P95-P5 en imagen; útil para seguimiento si la cámara permanece idéntica."},
            {"key":"bos_width_cm","label":"Ancho de base de sustentación estimado","value":bos_cm,"unit":"cm","quality":q if np.isfinite(bos_cm) else "Requiere escala/apoyo","notes":"Mediana entre centros de ambos pies durante doble apoyo 2D estimado."},
            {"key":"trendelenburg_drop_l_deg","label":"Caída pélvica dinámica lado izquierdo en suspensión","value":trend_l_swing,"unit":"°","quality":"Experimental" if support_reliable and np.isfinite(trend_l_swing) else "No fiable","notes":"Ángulo proyectado durante apoyo derecho; se anula si la segmentación temporal de apoyo no es consistente. No constituye por sí solo diagnóstico de Trendelenburg."},
            {"key":"trendelenburg_drop_r_deg","label":"Caída pélvica dinámica lado derecho en suspensión","value":trend_r_swing,"unit":"°","quality":"Experimental" if support_reliable and np.isfinite(trend_r_swing) else "No fiable","notes":"Ángulo proyectado durante apoyo izquierdo; se anula si la segmentación temporal de apoyo no es consistente. No constituye por sí solo diagnóstico de Trendelenburg."},
            {"key":"dynamic_knee_valgus_l_deg","label":"Valgo dinámico proyectado rodilla izquierda (P95)","value":valg_l,"unit":"°","quality":q,"notes":"Desviación medial proyectada respecto al eje cadera-tobillo; el valgo real es multiplanar."},
            {"key":"dynamic_knee_valgus_r_deg","label":"Valgo dinámico proyectado rodilla derecha (P95)","value":valg_r,"unit":"°","quality":q,"notes":"Desviación medial proyectada respecto al eje cadera-tobillo; el valgo real es multiplanar."},
            {"key":"trunk_pelvis_coupling_r","label":"Acoplamiento intersegmentario tronco-pelvis","value":coupling_r,"unit":"r","quality":q,"notes":"Correlación entre inclinación lateral del tronco y oblicuidad pélvica; descriptor 2D de estrategia de compensación."},
            {"key":"trunk_pelvis_phase_deg","label":"Desfase tronco-pelvis estimado","value":phase_deg,"unit":"° ciclo","quality":"Experimental" if np.isfinite(phase_deg) else "No calculable","notes":"v0.9.2: desfase de correlación cruzada normalizado al ciclo IC→IC y expresado entre −180° y +180°; útil para seguimiento, no diagnóstico aislado."},
        ]

    chart_data = {
        "frame": seg.frame.to_numpy(), "time_s": seg.frame.to_numpy()/fps,
        "Alternancia D-I": diff,
    }
    if is_frontal:
        chart_data.update({
            "Rodilla frontal izquierda": seg.L_frontal_knee_dev,
            "Rodilla frontal derecha": seg.R_frontal_knee_dev,
            "Orientación pie izquierda": seg.L_foot_progress_proj,
            "Orientación pie derecha": seg.R_foot_progress_proj,
            "Retropié izquierda": seg.L_rearfoot_tilt_proj,
            "Retropié derecha": seg.R_rearfoot_tilt_proj,
            "Oblicuidad pélvica": seg.pelvis_obliquity,
            "Oblicuidad de hombros": seg.shoulder_obliquity,
            "Inclinación lateral del tronco": seg.trunk_lateral_lean,
            "Relación hombros-pelvis": seg.shoulder_pelvis_rel,
            "CoM proxy lateral (px)": seg.com_proxy_x,
            "Valgo dinámico proyectado I": seg.L_dynamic_valgus,
            "Valgo dinámico proyectado D": seg.R_dynamic_valgus,
        })
    else:
        chart_data.update({
            "Rodilla izquierda": seg.L_knee_flex, "Rodilla derecha": seg.R_knee_flex,
            "Cadera izquierda": seg.L_hip_flex, "Cadera derecha": seg.R_hip_flex,
            "Tobillo izquierda": seg.L_ankle_angle, "Tobillo derecha": seg.R_ankle_angle,
            "Hombro izquierda": seg.L_shoulder_elev, "Hombro derecha": seg.R_shoulder_elev,
        })
    return metrics, pd.DataFrame(chart_data), seg

def metric_value(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m.get("value")
    return None


def metric_quality(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m.get("quality")
    return None


def fmt(v, n=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def biomech_summary(metrics, view="", prefix=""):
    """Resumen descriptivo 2D; evita convertir proxies en diagnósticos."""
    def mv(key):
        return metric_value(metrics, f"{prefix}{key}")
    parts = []
    n = mv("step_events_detected"); dur = mv("segment_duration_s"); cad = mv("cadence_exp")
    expected = mv("expected_steps_from_cadence"); err = mv("step_count_consistency_error_pct")
    if n is not None and dur is not None:
        txt = f"En el segmento de {fmt(dur,2)} s se detectaron {fmt(n,0)} eventos de paso/alternancia."
        if cad is not None and np.isfinite(cad):
            txt += f" La cadencia estimada fue {fmt(cad,1)} pasos/min"
            if expected is not None and np.isfinite(expected):
                txt += f", equivalente a {fmt(expected,1)} eventos esperados en esa duración"
            if err is not None and np.isfinite(err):
                txt += f" (discrepancia interna {fmt(err,1)} %)."
            else:
                txt += "."
        parts.append(txt)
    if "Frontal" in (view or ""):
        pkL, pkR = mv("frontal_knee_dev_l_p95"), mv("frontal_knee_dev_r_p95")
        pel = mv("pelvis_obliquity_rom"); sho = mv("shoulder_obliquity_rom"); tr = mv("trunk_lateral_lean_rom")
        if pkL is not None and pkR is not None:
            parts.append(f"El eje frontal de rodilla mostró P95 de {fmt(pkL,1)}° a izquierda y {fmt(pkR,1)}° a derecha. Son magnitudes proyectadas 2D, no una medición anatómica 3D de valgo/varo.")
        if pel is not None or sho is not None or tr is not None:
            parts.append(f"En el control axial/frontal, el ROM proyectado fue pelvis {fmt(pel,1)}°, hombros {fmt(sho,1)}° y tronco {fmt(tr,1)}°. Estas medidas describen oscilación y alineación en la imagen.")
        rfL, rfR = mv("rearfoot_tilt_l_p95"), mv("rearfoot_tilt_r_p95")
        if rfL is not None and rfR is not None:
            parts.append(f"La inclinación proyectada del retropié alcanzó P95 absoluto de {fmt(rfL,1)}° a izquierda y {fmt(rfR,1)}° a derecha. Puede informar sobre el patrón de inversión/eversión visible, pero no equivale a pronación 3D.")
    else:
        vals = []
        for k,label in [("hip_flex_diff_p95","cadera"),("knee_flex_diff_p95","rodilla"),("ankle_angle_diff_p95","tobillo"),("shoulder_elev_diff_p95","hombro")]:
            v=mv(k)
            if v is not None and np.isfinite(v): vals.append(f"{label} {fmt(v,1)}°")
        if vals:
            parts.append("Las diferencias D/I de P95 en la vista lateral fueron: " + ", ".join(vals) + ". Son diferencias 2D proyectadas y deben contextualizarse con el ciclo de marcha y la calidad del tracking.")
    return parts


def get_point(row, name, min_score=0.25):
    try:
        if row[f"{name}_score"] < min_score:
            return None
        return int(round(row[f"{name}_x"])), int(round(row[f"{name}_y"]))
    except Exception:
        return None


def render_angle_video(video_path, full_df, out_path, view="Lateral", assistive_device="Sin ayuda"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    enriched = add_angle_columns(full_df)
    if "Frontal" in (view or ""):
        enriched = add_frontal_columns(enriched)
    indexed = enriched.set_index("frame")
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no in indexed.index:
            row = indexed.loc[frame_no]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            for a,b in SKELETON:
                pa, pb = get_point(row,a), get_point(row,b)
                if pa and pb: cv2.line(frame, pa, pb, (40,220,80), 2, cv2.LINE_AA)
            for name in HALPE26:
                p = get_point(row,name)
                if p: cv2.circle(frame,p,3,(0,190,255),-1,cv2.LINE_AA)
            if "Frontal" in (view or ""):
                lines = [
                    f"Rod frontal I/D {row['L_frontal_knee_dev']:.0f}/{row['R_frontal_knee_dev']:.0f} deg",
                    f"Pie orient. I/D {row['L_foot_progress_proj']:.0f}/{row['R_foot_progress_proj']:.0f} deg",
                    f"Retropie I/D {row['L_rearfoot_tilt_proj']:.0f}/{row['R_rearfoot_tilt_proj']:.0f} deg",
                    f"Pelvis {row['pelvis_obliquity']:.0f} deg",
                    f"Hombros {row['shoulder_obliquity']:.0f} deg",
                    f"Tronco {row['trunk_lateral_lean']:.0f} deg",
                    "2D proyectado · no rotacion/pronacion 3D",
                ]
            else:
                lines = [
                    f"Cad I/D {row['L_hip_flex']:.0f}/{row['R_hip_flex']:.0f} deg",
                    f"Rod I/D {row['L_knee_flex']:.0f}/{row['R_knee_flex']:.0f} deg",
                    f"Tob I/D {row['L_ankle_angle']:.0f}/{row['R_ankle_angle']:.0f} deg",
                    f"Hom I/D {row['L_shoulder_elev']:.0f}/{row['R_shoulder_elev']:.0f} deg",
                    "2D proyectado",
                ]
            if assistive_device != "Sin ayuda":
                lines.append(f"Ayuda: {assistive_device}")
            y = 32
            for txt in lines:
                cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                y += 25
        writer.write(frame)
        frame_no += 1
    cap.release(); writer.release()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = f'"{ffmpeg}" -y -loglevel error -i "{raw}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out_path}"'
    rc = os.system(cmd)
    try: raw.unlink(missing_ok=True)
    except Exception: pass
    return out_path if rc == 0 and out_path.exists() else None


def cleanup_temp_session(session_dir):
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
        return True
    except Exception:
        return False

def _metric_sentence(metrics, key, title=None, digits=1):
    m=next((x for x in metrics if x.get("key")==key),None)
    if not m or m.get("value") is None:
        return None
    try:
        v=float(m["value"])
        if not np.isfinite(v): return None
    except Exception:
        return None
    label=title or m.get("label",key); unit=m.get("unit","")
    return f"{label}: {v:.{digits}f} {unit}".strip()+f" ({reference_text_for_metric(key)})"

def _fmt_metric_report(metrics, key, label=None, digits=1):
    v=metric_value(metrics,key)
    if v is None or not np.isfinite(v):
        return None
    m=next((x for x in metrics if x.get("key")==key),{})
    unit=m.get("unit","")
    return f"{label or m.get('label',key)}: {float(v):.{digits}f} {unit}".strip()+f" ({reference_text_for_metric(key)})"


def _phase_metric(metrics, base, two_cam=False):
    key=("lateral_"+base) if two_cam and metric_value(metrics,"lateral_"+base) is not None else base
    return key, metric_value(metrics,key)


def generate_reports(metrics, view, patient_code, record_name, assistive_device,
                     patient_age=0, patient_sex="No especificado", record_date=None):
    """Informe clínico estructurado v0.9.3 con metadatos editables, cierre temporal, CV robusto y trazabilidad multipersona."""
    two_cam = any(str(m.get("key", "")).startswith("lateral_") for m in metrics)
    tp = "lateral_" if two_cam else ""
    frontal_prefix = "front_" if two_cam else ""

    def val(key):
        return metric_value(metrics, key)

    def choose(base, prefer_frontal=False):
        candidates = []
        if prefer_frontal:
            candidates.append(frontal_prefix + base)
        candidates += [tp + base, base]
        for k in candidates:
            v = val(k)
            if v is not None and np.isfinite(v):
                return k, float(v)
        return candidates[-1], np.nan

    def ref_for(key):
        # Referencias redactadas para el informe; evita convertir medias/IC en umbrales diagnósticos.
        bare = key.replace("front_", "").replace("lateral_", "")
        if bare == "cadence_exp":
            return "Ref. poblacional contextual: 114.95–118.35 pasos/min (IC95% de la media de cadencia habitual al aire libre en adultos aparentemente sanos; Murtagh et al., Sports Med 2021, PMCID PMC7806575). No es un umbral diagnóstico individual."
        if bare == "regularity_cv":
            return "Sin umbral diagnóstico directamente transferible: en v0.9.2 este CV global es un promedio ponderado de los CV izquierdo y derecho calculados por separado sobre ciclos IC→IC rectilíneos validados y filtrados robustamente; no debe compararse directamente con umbrales publicados para CV de tiempo de paso."
        if bare == "temporal_asymmetry_exp":
            return "Sin umbral universal directamente transferible a este detector 2D. En laboratorio la simetría temporal sana suele aproximarse a 1:1; una cohorte de mujeres jóvenes activas mostró baja asimetría a velocidad preferida (PMCID PMC6335661)."
        if bare in {"stance_time_l_2d", "stance_time_r_2d", "stance_asymmetry_2d"}:
            return "Ref. contextual: el apoyo sano es aproximadamente simétrico entre lados; esta es una estimación cinemática 2D experimental, no una medida de plataforma de fuerzas."
        if bare in {"step_events_detected", "segment_duration_s"}:
            return "Referencia metodológica: control de consistencia interna del propio registro; no existe rango normativo clínico aplicable."
        if bare in {"trendelenburg_drop_l_deg", "trendelenburg_drop_r_deg"}:
            return "Sin umbral diagnóstico 2D universal validado para HALPE26. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente."
        if bare in {"dynamic_knee_valgus_l_deg", "dynamic_knee_valgus_r_deg"}:
            return "Sin rango normativo validado específicamente para HALPE26 2D. Descriptor de desviación medial proyectada; el valgo dinámico real es multiplanar."
        if bare == "trunk_pelvis_coupling_r":
            return "Sin banda normativa universal para este coeficiente 2D. r≈+1 indica acoplamiento lineal en fase; r≈−1, contrafase; el significado depende de la tarea y del contexto clínico."
        if bare == "trunk_pelvis_phase_deg":
            return "Sin banda normativa universal. En v0.9.2 el desfase se expresa de forma circular entre −180° y +180°; interpretar longitudinalmente y junto con la calidad del tracking."
        return "Sin umbral normativo 2D validado directamente transferible a esta métrica; descriptor proyectado que requiere contextualización clínica."

    # -------- datos principales --------
    kcad, cad = choose("cadence_exp")
    kcv, cv = choose("regularity_cv")
    kasym, asym = choose("temporal_asymmetry_exp")
    kn, n_events = choose("step_events_detected")
    kdur, duration = choose("segment_duration_s")
    ksl, stance_l = choose("stance_time_l_2d")
    ksr, stance_r = choose("stance_time_r_2d")
    ksa, stance_asym = choose("stance_asymmetry_2d")
    _, support_score_l = choose("support_segmentation_score_l")
    _, support_score_r = choose("support_segmentation_score_r")
    support_ok = np.isfinite(stance_l) and np.isfinite(stance_r)

    lines = []
    lines.append("INFORME DE ANÁLISIS BIOMECÁNICO DE LA MARCHA (2D)")
    lines.append("")
    lines.append("Ficha del registro:")
    lines.append(f"• Paciente / Código: {patient_code}")
    if patient_age is not None and int(patient_age or 0) > 0:
        lines.append(f"• Edad: {int(patient_age)} años")
    if patient_sex and str(patient_sex) != "No especificado":
        lines.append(f"• Sexo: {patient_sex}")
    if record_date:
        try:
            date_txt = record_date.strftime("%d/%m/%Y")
        except Exception:
            date_txt = str(record_date)
        lines.append(f"• Fecha del registro: {date_txt}")
    if record_name:
        lines.append(f"• Registro: {record_name}")
    lines.append(f"• Prueba: Análisis de marcha en vista {view}")
    lines.append(f"• Ayuda técnica: {assistive_device}")
    if np.isfinite(n_events) and np.isfinite(duration):
        lines.append(f"• Consistencia interna: {n_events:.0f} eventos detectados en {duration:.2f} s de registro analizado ({ref_for(kn)})")
    else:
        lines.append("• Consistencia interna: no calculable con los datos disponibles (Referencia metodológica: control interno del registro; no existe rango normativo clínico aplicable).")
    if np.isfinite(support_score_l) and np.isfinite(support_score_r):
        qtmp = "suficiente para estimación experimental" if support_ok else "insuficiente: stance/swing anulados"
        lines.append(f"• Control de segmentación apoyo/oscilación: izquierda {support_score_l:.0f}/100; derecha {support_score_r:.0f}/100 — {qtmp} (control matemático interno, no escala clínica).")
    coh_flag = val("temporal_coherence_flag")
    coh_err = val("temporal_coherence_max_error_pct")
    if coh_flag is not None and np.isfinite(coh_flag):
        if coh_flag >= 0.5:
            err_txt = f"{coh_err:.1f}%" if coh_err is not None and np.isfinite(coh_err) else "dentro del límite"
            lines.append(f"• Coherencia temporal v0.9.2: SUPERADA — discordancia máxima {err_txt}. Cadencia, asimetría y doble apoyo pueden publicarse si superan además sus controles específicos.")
        else:
            err_txt = f"{coh_err:.1f}%" if coh_err is not None and np.isfinite(coh_err) else "no cuantificable"
            lines.append(f"• Coherencia temporal v0.9.2: NO SUPERADA — discordancia máxima {err_txt}. El doble apoyo incompatible se suprime. Cadencia, CV y asimetría temporal pueden mantenerse mediante el detector cinemático distal cuando existe periodicidad articular suficiente, siempre con su calidad explícita.")

    # v0.9.0 · trazabilidad del sujeto cuando existe acompañante/terapeuta.
    manual_keys = ["subject_manual_selection_flag","front_subject_manual_selection_flag","lateral_subject_manual_selection_flag"]
    manual_selected = any((val(k) is not None and np.isfinite(val(k)) and val(k) >= 0.5) for k in manual_keys)
    if manual_selected:
        lines.append("• Selección de sujeto: paciente seleccionado manualmente y seguimiento de identidad bloqueado; las demás personas detectadas quedan excluidas del análisis.")
        cont_candidates = [val("identity_continuity_pct"), val("front_identity_continuity_pct"), val("lateral_identity_continuity_pct")]
        cont_candidates = [float(x) for x in cont_candidates if x is not None and np.isfinite(x)]
        amb_candidates = [val("identity_ambiguous_excluded_pct"), val("front_identity_ambiguous_excluded_pct"), val("lateral_identity_ambiguous_excluded_pct")]
        amb_candidates = [float(x) for x in amb_candidates if x is not None and np.isfinite(x)]
        if cont_candidates:
            lines.append(f"• Continuidad de identidad del sujeto: {min(cont_candidates):.1f}% como valor conservador entre vistas.")
        if amb_candidates:
            lines.append(f"• Frames ambiguos excluidos por riesgo de cambio de identidad: {max(amb_candidates):.1f}% como valor conservador entre vistas.")

    lines.append("")
    lines.append("1. PARÁMETROS ESPACIOTEMPORALES (RITMO, VARIABILIDAD Y CARGA)")
    if np.isfinite(cad):
        lines.append(f"• Cadencia estimada: {cad:.1f} pasos/min ({ref_for(kcad)})")
    if np.isfinite(cv):
        lines.append(f"• Variabilidad temporal (CV): {cv:.1f}% ({ref_for(kcv)})")
    if np.isfinite(asym):
        lines.append(f"• Asimetría temporal global: {asym:.1f}% ({ref_for(kasym)})")

    if np.isfinite(stance_l) and np.isfinite(stance_r):
        if stance_l > stance_r:
            dominant = "IZQUIERDA"
        elif stance_r > stance_l:
            dominant = "DERECHA"
        else:
            dominant = "SIMILAR EN AMBAS EXTREMIDADES"
        sa_txt = f"{stance_asym:.1f}%" if np.isfinite(stance_asym) else "no calculable"
        lines.append("• Tiempo de apoyo 2D estimado (fase de carga):")
        lines.append(f"  - Izquierda: {stance_l:.2f} s ({ref_for(ksl)})")
        lines.append(f"  - Derecha: {stance_r:.2f} s ({ref_for(ksr)})")
        lines.append(f"  - Asimetría de carga: {sa_txt}, con mayor tiempo de apoyo estimado en la extremidad {dominant} ({stance_l:.2f} s vs. {stance_r:.2f} s) ({ref_for(ksa)})")
    else:
        lines.append("• Tiempo de apoyo 2D estimado: NO INFORMADO. La segmentación apoyo/oscilación no supera el control de consistencia temporal en ambos lados; se evita publicar una duración potencialmente falsa (estimación markerless 2D, no plataforma de fuerzas).")

    # Opcional doble apoyo si está disponible, siempre con referencia.
    kds, ds = choose("double_support_pct_2d")
    if np.isfinite(ds):
        lines.append(f"• Apoyo bipodal estimado: {ds:.1f}% del segmento/ciclo analizado (Ref. contextual: en marcha adulta habitual el doble apoyo total suele ocupar aproximadamente 20–24% del ciclo; la estimación v0.9.2 usa el mismo dominio de ciclos rectilíneos que cadencia/apoyo y se anula si falla el control físico interno).")

    lines.append("")
    lines.append("2. CINEMÁTICA PROYECTADA 2D (ESTABILIDAD FRONTAL Y APOYO MONOPODAL)")
    frontal_available = ("Frontal" in (view or "")) or any(str(m.get("key", "")).startswith("front_") for m in metrics)
    if frontal_available:
        kpl, drop_l = choose("trendelenburg_drop_l_deg", prefer_frontal=True)
        kpr, drop_r = choose("trendelenburg_drop_r_deg", prefer_frontal=True)
        kvl, valg_l = choose("dynamic_knee_valgus_l_deg", prefer_frontal=True)
        kvr, valg_r = choose("dynamic_knee_valgus_r_deg", prefer_frontal=True)
        lines.append("• Caída pélvica en suspensión (drop pélvico durante apoyo monopodal):")
        lines.append(f"  - Lado Izquierdo: {drop_l:.1f}° ({ref_for(kpl)})" if np.isfinite(drop_l) else "  - Lado Izquierdo: no calculable (Sin umbral diagnóstico 2D universal; confirmar clínicamente).")
        lines.append(f"  - Lado Derecho: {drop_r:.1f}° ({ref_for(kpr)})" if np.isfinite(drop_r) else "  - Lado Derecho: no calculable (Sin umbral diagnóstico 2D universal; confirmar clínicamente).")
        lines.append("• Valgo dinámico proyectado (desviación medial durante carga):")
        lines.append(f"  - Lado Izquierdo: {valg_l:.1f}° ({ref_for(kvl)})" if np.isfinite(valg_l) else "  - Lado Izquierdo: no calculable (Sin rango normativo validado específicamente para HALPE26 2D).")
        lines.append(f"  - Lado Derecho: {valg_r:.1f}° ({ref_for(kvr)})" if np.isfinite(valg_r) else "  - Lado Derecho: no calculable (Sin rango normativo validado específicamente para HALPE26 2D).")
        lines.append("• Precaución metodológica: el valgo/varo representa únicamente la desviación medial/lateral proyectada en el plano frontal y no una medición 3D multiplanar aislada. El drop pélvico es un descriptor 2D durante apoyo monopodal y debe confirmarse mediante exploración clínica cuando tenga relevancia terapéutica.")
    else:
        lines.append("• La vista actual no permite una estimación frontal fiable de drop pélvico o valgo/varo proyectado (Sin umbral normativo 2D aplicable porque la geometría de captura no es comparable).")

    lines.append("")
    lines.append("3. COORDINACIÓN INTERSEGMENTARIA (TRONCO-PELVIS)")
    if frontal_available:
        kcr, coupling = choose("trunk_pelvis_coupling_r", prefer_frontal=True)
        kph, phase = choose("trunk_pelvis_phase_deg", prefer_frontal=True)
        if np.isfinite(coupling):
            lines.append(f"• Acoplamiento tronco-pelvis (r): {coupling:.2f} r ({ref_for(kcr)})")
        else:
            lines.append("• Acoplamiento tronco-pelvis (r): no calculable (Sin banda normativa universal para este descriptor 2D).")
        if np.isfinite(phase):
            lines.append(f"• Desfase tronco-pelvis: {phase:.1f}° de ciclo ({ref_for(kph)})")
        else:
            lines.append("• Desfase tronco-pelvis: no calculable (Sin banda normativa universal; 360° = 1 ciclo de marcha).")
        lines.append("• Interpretación: r próximo a +1 indica que tronco y pelvis oscilan de forma linealmente similar/en fase; r próximo a −1 sugiere contrafase. El desfase cuantifica el retraso/adelanto relativo en grados de ciclo (360° = 1 ciclo completo) y puede ayudar a describir estrategias compensatorias, pero no diagnostica por sí solo un patrón tipo Duchenne.")
    else:
        lines.append("• La coordinación frontal tronco-pelvis no se estima de forma fiable desde una única vista lateral (Sin referencia normativa aplicable).")

    lines.append("")
    lines.append("4. IMPRESIÓN BIOMECÁNICA GENERAL")
    synthesis = []
    if np.isfinite(cad):
        synthesis.append(f"El registro presenta una cadencia estimada de {cad:.1f} pasos/min, que debe interpretarse frente a la referencia poblacional sana solo como contexto y no como objetivo terapéutico automático")
    if np.isfinite(stance_l) and np.isfinite(stance_r):
        dom = "izquierda" if stance_l > stance_r else "derecha" if stance_r > stance_l else "sin predominio lateral claro"
        synthesis.append(f"la estimación temporal por ciclos continuos muestra mayor permanencia en apoyo en la extremidad {dom} ({stance_l:.2f} s izquierda vs. {stance_r:.2f} s derecha)")
    else:
        synthesis.append("el tiempo de apoyo y su direccionalidad no se informan porque la segmentación temporal no alcanzó consistencia suficiente")
    if frontal_available:
        bits=[]
        if 'drop_l' in locals() and np.isfinite(drop_l): bits.append(f"drop pélvico izquierdo {drop_l:.1f}°")
        if 'drop_r' in locals() and np.isfinite(drop_r): bits.append(f"derecho {drop_r:.1f}°")
        if 'valg_l' in locals() and np.isfinite(valg_l): bits.append(f"valgo proyectado izquierdo {valg_l:.1f}°")
        if 'valg_r' in locals() and np.isfinite(valg_r): bits.append(f"derecho {valg_r:.1f}°")
        if bits:
            synthesis.append("en el plano frontal se observan " + ", ".join(bits) + ", todos ellos descriptores 2D sin umbral diagnóstico universal")
        if 'coupling' in locals() and np.isfinite(coupling):
            ptxt = f" y desfase {phase:.1f}°" if 'phase' in locals() and np.isfinite(phase) else ""
            synthesis.append(f"la coordinación tronco-pelvis muestra r={coupling:.2f}{ptxt}, útil principalmente para seguimiento longitudinal y contextualización clínica")
    if not synthesis:
        lines.append("No hay suficientes métricas válidas para generar una síntesis biomecánica cuantitativa. Debe revisarse la calidad del tracking y el intervalo seleccionado.")
    else:
        # 3–4 líneas cortas, unificadas y sin diagnóstico automático.
        lines.append(". ".join(synthesis) + ".")
        lines.append("Los hallazgos deben integrarse con velocidad de marcha, ayuda técnica, exploración neurológica/musculoesquelética y evolución intraindividual; las métricas markerless 2D no sustituyen una evaluación 3D o instrumentada cuando la decisión clínica depende de fuerzas, contacto o cinemática multiplanar.")

    # La salida principal solicitada es el informe clínico. Mantener una versión paciente separada
    # para compatibilidad con la interfaz anterior, pero no mezclarla dentro de los 4 bloques obligatorios.
    patient=[]
    patient.append("VERSIÓN SIMPLIFICADA PARA EL PACIENTE")
    if np.isfinite(cad):
        patient.append(f"En este registro se estimaron {cad:.1f} pasos por minuto. Las cifras de personas sanas se muestran solo como referencia general; en rehabilitación importa especialmente cómo cambia tu marcha respecto a tus propios registros.")
    if np.isfinite(stance_l) and np.isfinite(stance_r):
        side = "izquierda" if stance_l > stance_r else "derecha" if stance_r > stance_l else "ambas de forma parecida"
        patient.append(f"El análisis sugiere que la pierna que permanece apoyada durante más tiempo es la {side}. La estimación utiliza ciclos continuos y se interpreta junto con la exploración clínica.")
    else:
        patient.append("En este registro no se ofrece un tiempo de apoyo por pierna porque el vídeo no permitió segmentar apoyo y oscilación con suficiente consistencia. Es preferible no mostrar una cifra dudosa.")
    if frontal_available:
        patient.append("También se observa cómo se mantiene la pelvis cuando una pierna está en el aire, cómo se alinean las rodillas de frente y cómo se coordinan el tronco y la pelvis. Son medidas proyectadas en una imagen 2D, útiles sobre todo para comparar tu evolución en sesiones realizadas de forma similar.")
    patient.append("El resultado no se interpreta de forma aislada: se combina con tus síntomas, capacidad funcional, ayudas utilizadas y evolución durante el tratamiento.")
    return "\n".join(lines), "\n\n".join(patient)

# ------------------------- UI -------------------------
# ------------------------- UI -------------------------
st.title("PhysioSentinel Gait")
st.caption(f"Versión {APP_VERSION} · selección multipersona con identidad bloqueada · 2 cámaras · preparación 3D · Supabase tolerante a fallos")

if sb_ready():
    st.success("☁️ Supabase conectado: pacientes, sesiones, métricas y perfiles de calibración pueden persistir. Los vídeos NO se guardan.")
else:
    st.error("Supabase aún no está configurado. Añade SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY en Streamlit Secrets.")

# Estado base
for k, v in {
    "pose_done": False, "metrics_done": False, "temp_deleted": False,
    "analysis_df": None, "analysis_df2": None,
    "annotated_video_bytes": None, "annotated_video2_bytes": None,
    "sync_offset_auto_s": 0.0, "sync_offset_user_s": 0.0,
    "sync_correlation": np.nan, "sync_quality": "No calculable",
    "camera1_quality": None, "camera2_quality": None,
    "pose_raw_done": False, "subject_locked": False,
    "subject_selection_cam1": None, "subject_selection_cam2": None,
    "subject_preview_cam1": None, "subject_preview_cam2": None,
    "tracking_info_cam1": None, "tracking_info_cam2": None,
    "pose_json1": None, "pose_json2": None,
    # v0.9.3 · metadatos editables sin reiniciar/reprocesar el análisis
    "patient_code": "Prueba",
    "record_name": "Marcha",
    "patient_age": 0,
    "patient_sex": "No especificado",
    "record_date": datetime.now().date(),
    # v0.10.1 · snapshot ligero para el analizador de ciclo
    "cycle_seg": None,
    "cycle_seg_front": None,
    "cycle_seg_lateral": None,
    "cycle_fps": None,
    "cycle_fps_front": None,
    "cycle_fps_lateral": None,
    "cycle_support_left": None,
    "cycle_support_right": None,
    "cycle_support_left_front": None,
    "cycle_support_right_front": None,
    "cycle_support_left_lateral": None,
    "cycle_support_right_lateral": None,
    "cycle_source_view": None,
    # v0.10.3 · vídeo limpio de sesión para pestaña 9 (no Supabase)
    "cycle_video_clean_bytes": None,
    "cycle_video2_clean_bytes": None,
    "selected_subject_label_cam1": None,
    "selected_subject_label_cam2": None,
    "cycle_phase_review_base": None,
    "cycle_phase_review_bytes": None,
    "cycle_phase_review_speed": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Perfiles de calibración disponibles
calibration_names = []
if sb_ready():
    try:
        calibration_names = [x["name"] for x in sb_list_calibrations()]
    except Exception:
        calibration_names = []

with st.sidebar:
    st.header("Sesión")

    # v0.9.3 · Los datos personales se editan dentro de un formulario.
    # Escribir en los campos NO provoca reruns sucesivos. Solo el botón Guardar
    # actualiza los metadatos y nunca invalida vídeo, pose, métricas ni segmento.
    with st.expander("👤 Datos del paciente / registro", expanded=True):
        with st.form("patient_metadata_form", clear_on_submit=False):
            patient_edit = st.text_input(
                "Paciente / código",
                value=str(st.session_state.get("patient_code", "Prueba")),
            )
            record_edit = st.text_input(
                "Nombre del registro",
                value=str(st.session_state.get("record_name", "Marcha")),
            )
            age_edit = st.number_input(
                "Edad",
                min_value=0, max_value=120,
                value=int(st.session_state.get("patient_age", 0) or 0),
                step=1,
                help="0 = no especificada.",
            )
            sex_options = ["No especificado", "Mujer", "Hombre", "Otro / no binario"]
            current_sex = str(st.session_state.get("patient_sex", "No especificado"))
            sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
            sex_edit = st.selectbox("Sexo", sex_options, index=sex_index)
            date_edit = st.date_input(
                "Fecha del registro",
                value=st.session_state.get("record_date", datetime.now().date()),
            )
            save_metadata = st.form_submit_button(
                "💾 Guardar datos",
                type="primary",
                use_container_width=True,
            )

        if save_metadata:
            old_code = str(st.session_state.get("patient_code", ""))
            st.session_state["patient_code"] = patient_edit.strip() or "Prueba"
            st.session_state["record_name"] = record_edit.strip() or "Marcha"
            st.session_state["patient_age"] = int(age_edit)
            st.session_state["patient_sex"] = sex_edit
            st.session_state["record_date"] = date_edit

            # Regenerar textos del informe con los nuevos metadatos,
            # sin tocar resultados ni archivos de pose.
            st.session_state.pop("report_technical", None)
            st.session_state.pop("report_patient", None)

            # Si la sesión ya existe en Supabase, reasignar código y nombre
            # sin crear una sesión nueva ni reprocesar el vídeo.
            sid = st.session_state.get("cloud_session_id")
            if sid and sb_ready():
                try:
                    sb_update_session_identity(
                        sid,
                        patient_code=st.session_state["patient_code"],
                        record_name=st.session_state["record_name"],
                    )
                except Exception as e:
                    st.warning(f"Los datos se han actualizado en esta sesión, pero Supabase no pudo sincronizar código/nombre del registro: {e}")

            st.success("Datos actualizados. El análisis biomecánico se conserva intacto.")

    patient = str(st.session_state.get("patient_code", "Prueba"))
    record = str(st.session_state.get("record_name", "Marcha"))
    patient_age = int(st.session_state.get("patient_age", 0) or 0)
    patient_sex = str(st.session_state.get("patient_sex", "No especificado"))
    record_date = st.session_state.get("record_date", datetime.now().date())

    st.caption("Usa preferentemente un código seudonimizado en lugar de nombre y apellidos.")
    st.divider()
    mode = st.radio(
        "Modo de análisis",
        ["1 cámara · 2D", "2 cámaras · frontal/posterior + lateral · preparación 3D"],
        index=0,
    )
    is_two_cam = mode.startswith("2 cámaras")
    view = st.radio("Vista", ["Frontal/posterior", "Lateral"], index=0) if not is_two_cam else "Frontal+Lateral"
    frontal_orientation = st.selectbox(
        "Sentido de la toma frontal/posterior",
        ["No especificada", "Frontal", "Posterior", "Mixta/ida-vuelta"],
        index=0,
    ) if "Frontal" in view else "No aplica"
    assistive_device = st.selectbox("Ayuda técnica", ASSISTIVE_OPTIONS, index=0)
    if assistive_device != "Sin ayuda":
        st.caption("La app medirá la visibilidad por regiones y marcará métricas condicionadas por posibles oclusiones.")
    subject_mode = st.radio(
        "Sujeto biomecánico",
        ["Automático · una sola persona", "Selección manual · multipersona"],
        index=0,
        help="Usa selección manual cuando el paciente camina acompañado, asistido o estrechamente supervisado."
    )
    manual_subject_mode = subject_mode.startswith("Selección manual")
    if manual_subject_mode:
        st.caption("v0.9.3: primero se detectan las personas; tú eliges al paciente y después la identidad queda bloqueada. Los frames ambiguos se excluyen.")
    if is_two_cam:
        opts = ["Sin calibración"] + calibration_names
        default_cal = st.session_state.get("selected_calibration_name") or "Sin calibración"
        try:
            idx = opts.index(default_cal)
        except ValueError:
            idx = 0
        selected_calibration = st.selectbox("Perfil de calibración 3D", opts, index=idx)
        st.session_state.selected_calibration_name = None if selected_calibration == "Sin calibración" else selected_calibration
    else:
        selected_calibration = "Sin calibración"
        st.session_state.selected_calibration_name = None
    st.divider()
    st.markdown("**Escala espacial frontal (opcional)**")
    scale_cm_per_px = st.number_input("cm por píxel", min_value=0.0, value=float(st.session_state.get("scale_cm_per_px",0.0)), step=0.001, format="%.4f", help="Solo si dispones de una calibración espacial válida en el plano de marcha. 0 = sin escala métrica.")
    st.session_state.scale_cm_per_px=float(scale_cm_per_px)
    st.caption("Sin escala válida, CoM y BoS se mantienen como proxies relativos/píxel y no se convierten a cm.")
    st.divider()
    st.markdown("**Motor interno**")
    st.write("Pose2Sim + RTMPose")
    st.write("Body_with_feet / HALPE26")
    st.caption("Vídeos → /tmp → pose 2D → sincronización → resultados → Supabase → eliminación")


tabs = st.tabs([
    "1 · Vídeos", "2 · Calidad", "3 · Analizar marcha", "4 · Resultados 2D",
    "5 · Pacientes / Evolución", "6 · Redacción informe", "7 · 3D / Calibración",
    "8 · Exportar / Descargar", "9 · Ciclo de marcha"
])

with tabs[0]:
    st.subheader("Carga temporal de vídeo")
    st.info("Los vídeos se usan únicamente para este análisis. No se suben a Supabase ni quedan guardados en el histórico.")
    if not is_two_cam:
        up1 = st.file_uploader(f"Vídeo {view.lower()}", type=["mp4","mov","avi","mkv"], key="uploader_video1")
        if up1:
            st.video(up1)
        up2 = None
    else:
        st.markdown("### Protocolo 2 cámaras")
        st.write("Cámara 1: frontal/posterior. Cámara 2: lateral. Ambas deben grabar simultáneamente la misma zona de marcha.")
        st.caption("Para facilitar la sincronización, realiza al inicio un evento corporal breve y visible en ambas cámaras (por ejemplo, una elevación vertical rápida) antes de iniciar la marcha.")
        c1, c2 = st.columns(2)
        with c1:
            up1 = st.file_uploader("Cámara 1 · frontal/posterior", type=["mp4","mov","avi","mkv"], key="uploader_front")
            if up1:
                st.video(up1)
        with c2:
            up2 = st.file_uploader("Cámara 2 · lateral", type=["mp4","mov","avi","mkv"], key="uploader_side")
            if up2:
                st.video(up2)
        st.caption("v0.7 procesa RTMPose en las dos cámaras, calcula 2D complementario por vista y estima el desfase temporal. La triangulación 3D todavía no se ejecuta automáticamente.")

    if st.button("Crear sesión temporal", type="primary", use_container_width=True):
        if up1 is None or (is_two_cam and up2 is None):
            st.error("Selecciona el/los vídeo(s) necesarios.")
        else:
            try:
                folder = create_temp_session(patient, record)
                p1 = folder / "videos" / f"cam01{Path(up1.name).suffix.lower() or '.mp4'}"
                save_upload(up1, p1)
                meta1 = video_metadata(p1)
                if not meta1:
                    raise RuntimeError("No puedo leer el vídeo de cámara 1.")
                p2 = None
                meta2 = None
                if up2 is not None:
                    p2 = folder / "videos" / f"cam02{Path(up2.name).suffix.lower() or '.mp4'}"
                    save_upload(up2, p2)
                    meta2 = video_metadata(p2)
                    if not meta2:
                        raise RuntimeError("No puedo leer el vídeo de cámara 2.")
                st.session_state.update({
                    "session_dir": str(folder), "video1_path": str(p1), "video2_path": str(p2) if p2 else None,
                    "meta1": meta1, "meta2": meta2, "mode": mode, "view": view,
                    "subject_mode": subject_mode,
                    "pose_done": False, "pose_raw_done": False, "subject_locked": False,
                    "subject_selection_cam1": None, "subject_selection_cam2": None,
                    "subject_preview_cam1": None, "subject_preview_cam2": None,
                    "tracking_info_cam1": None, "tracking_info_cam2": None,
                    "metrics_done": False, "temp_deleted": False,
                    "analysis_df": None, "analysis_df2": None,
                    "annotated_video_bytes": None, "annotated_video2_bytes": None,
                    "patient_code": st.session_state.get("patient_code", patient.strip()),
                    "record_name": st.session_state.get("record_name", record.strip()),
                    "patient_age": int(st.session_state.get("patient_age", 0) or 0),
                    "patient_sex": st.session_state.get("patient_sex", "No especificado"),
                    "record_date": st.session_state.get("record_date", datetime.now().date()),
                    "assistive_device": assistive_device, "frontal_orientation": frontal_orientation,
                    "sync_offset_auto_s": 0.0, "sync_offset_user_s": 0.0,
                    "sync_correlation": np.nan, "sync_quality": "No calculable",
                    "camera1_quality": None, "camera2_quality": None,
                    "cloud_session_id": None,
                    "selected_calibration_name": st.session_state.get("selected_calibration_name"),
                })
                cloud_ok = False
                if sb_ready():
                    try:
                        sid = sb_create_session(
                            st.session_state.get("patient_code", patient.strip()),
                            st.session_state.get("record_name", record.strip()),
                            mode, view, meta1,
                            assistive_device, frontal_orientation,
                            meta2=meta2,
                            calibration_profile_name=st.session_state.get("selected_calibration_name"),
                        )
                        st.session_state["cloud_session_id"] = sid
                        cloud_ok = True
                    except Exception as sb_e:
                        st.session_state["cloud_session_id"] = None
                        st.warning(f"La sesión temporal se ha creado, pero Supabase no pudo registrar esta sesión: {sb_e}")
                if cloud_ok:
                    st.success("Sesión temporal creada. Supabase conserva la sesión y su contexto; los vídeos permanecen solo en /tmp.")
                else:
                    st.success("Sesión temporal creada. Puedes analizar y ver resultados aunque Supabase no esté disponible.")
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Control de calidad de captura")
    meta1 = st.session_state.get("meta1")
    meta2 = st.session_state.get("meta2")
    if not meta1:
        st.info("Crea primero una sesión temporal.")
    else:
        if st.session_state.get("mode", mode).startswith("2 cámaras") and meta2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Cámara 1 · frontal/posterior")
                st.metric("FPS", f"{meta1['fps']:.2f}")
                st.metric("Duración", f"{meta1['duration']:.2f} s")
                st.write(f"Resolución: **{meta1['width']} × {meta1['height']}**")
            with c2:
                st.markdown("### Cámara 2 · lateral")
                st.metric("FPS", f"{meta2['fps']:.2f}")
                st.metric("Duración", f"{meta2['duration']:.2f} s")
                st.write(f"Resolución: **{meta2['width']} × {meta2['height']}**")
            fps_diff = abs(meta1["fps"] - meta2["fps"])
            dur_diff = abs(meta1["duration"] - meta2["duration"])
            a, b = st.columns(2)
            a.metric("Diferencia FPS", f"{fps_diff:.3f}")
            b.metric("Diferencia de duración", f"{dur_diff:.2f} s")
            if fps_diff <= 0.5:
                st.success("Las frecuencias de imagen son suficientemente próximas para la preparación 3D.")
            else:
                st.warning("Las cámaras tienen FPS diferentes. Se puede analizar 2D, pero la reconstrucción 3D requerirá remuestreo/sincronización cuidadosa.")
            if dur_diff > 2.0:
                st.warning("La duración difiere más de 2 s. Revisa que ambas grabaciones cubran el mismo ensayo.")
        else:
            a,b,c,d = st.columns(4)
            a.metric("FPS", f"{meta1['fps']:.1f}")
            b.metric("Duración", f"{meta1['duration']:.1f} s")
            c.metric("Resolución", f"{meta1['width']} × {meta1['height']}")
            d.metric("Orientación", meta1['orientation'])
        if st.session_state.get("assistive_device", "Sin ayuda") != "Sin ayuda":
            st.info(f"Marcha con ayuda técnica: **{st.session_state.get('assistive_device')}**. Tras RTMPose se calculará la visibilidad por cámara y región corporal.")

with tabs[2]:
    st.subheader("Analizar marcha")
    if not st.session_state.get("session_dir"):
        st.info("Crea primero una sesión temporal.")
    elif st.session_state.get("temp_deleted"):
        st.info("Los archivos temporales ya fueron eliminados después de guardar los resultados.")
    else:
        current_two = st.session_state.get("mode", "").startswith("2 cámaras")
        manual_mode = str(st.session_state.get("subject_mode","")).startswith("Selección manual")
        st.write("Motor: **Pose2Sim + RTMPose · Body_with_feet (HALPE26)**")
        if manual_mode:
            st.info("**Modo multipersona v0.9.1:** la app detectará las personas, pero NO decidirá cuál es el paciente. Después de la detección deberás seleccionarlo explícitamente.")
        if current_two:
            st.write("Se realizará detección de pose en **cam01 frontal/posterior y cam02 lateral** y después se estimará su desfase temporal.")

        if not st.session_state.get("pose_raw_done") and st.button("▶ Detectar pose" if manual_mode else "▶ Analizar marcha", type="primary", use_container_width=True):
            try:
                session_dir = Path(st.session_state.session_dir)
                with st.spinner("Detectando pose con Pose2Sim/RTMPose. En 2 cámaras el proceso tarda aproximadamente el doble que una sola vista..."):
                    cfg = prepare_config(session_dir)
                    run_pose2sim(cfg)
                json1 = find_pose_json_dir(session_dir, "cam01")
                if not json1:
                    raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON de cam01.")
                st.session_state.pose_json1 = str(json1)

                json2 = None
                if current_two:
                    json2 = find_pose_json_dir(session_dir, "cam02")
                    if not json2:
                        raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON de cam02.")
                    st.session_state.pose_json2 = str(json2)

                if manual_mode:
                    sel1 = scan_subject_candidates(json1)
                    if not sel1 or not sel1.get("candidates"):
                        raise RuntimeError("No encuentro ninguna persona suficientemente visible en cam01 para realizar la selección manual.")
                    st.session_state.subject_selection_cam1 = sel1
                    st.session_state.subject_preview_cam1 = render_subject_preview(st.session_state.video1_path, sel1)

                    if current_two and json2:
                        sel2 = scan_subject_candidates(json2)
                        if not sel2 or not sel2.get("candidates"):
                            raise RuntimeError("No encuentro ninguna persona suficientemente visible en cam02 para realizar la selección manual.")
                        st.session_state.subject_selection_cam2 = sel2
                        st.session_state.subject_preview_cam2 = render_subject_preview(st.session_state.video2_path, sel2)

                    st.session_state.pose_raw_done = True
                    st.session_state.pose_done = False
                    st.success("Pose detectada. Selecciona ahora el paciente en la imagen y bloquea su identidad.")
                else:
                    df1 = load_pose_dataframe(json1)
                    if df1.empty:
                        raise RuntimeError("No se pudieron leer keypoints HALPE26 de cam01.")
                    st.session_state.analysis_df = df1
                    q1 = camera_pose_quality(df1)
                    st.session_state.camera1_quality = q1

                    if current_two:
                        df2 = load_pose_dataframe(json2)
                        if df2.empty:
                            raise RuntimeError("No se pudieron leer keypoints HALPE26 de cam02.")
                        st.session_state.analysis_df2 = df2
                        q2 = camera_pose_quality(df2)
                        st.session_state.camera2_quality = q2
                        meta1 = st.session_state.meta1; meta2 = st.session_state.meta2
                        off, corr, sq = estimate_sync_offset(df1, meta1["fps"], df2, meta2["fps"], max_offset_s=2.0)
                        st.session_state.sync_offset_auto_s = off
                        st.session_state.sync_offset_user_s = off
                        st.session_state.sync_correlation = corr
                        st.session_state.sync_quality = sq
                        ready, reasons = readiness_3d(st.session_state.mode, q1, q2, sq, st.session_state.get("selected_calibration_name"))
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            sb_update_session_3d(
                                st.session_state.cloud_session_id,
                                sync_offset_s=off,
                                sync_correlation=corr if np.isfinite(corr) else None,
                                sync_quality=sq,
                                calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                ready_3d=bool(ready),
                            )
                        st.success(f"Pose completada en dos cámaras: cam01 {len(df1)} frames útiles · cam02 {len(df2)} frames útiles.")
                        st.info(f"Sincronización automática experimental: **{off:+.3f} s** · correlación **{corr:.2f}** · calidad **{sq}**.")
                    else:
                        st.session_state.analysis_df2 = None
                        st.success(f"Pose completada: {len(df1)} frames útiles.")
                    st.session_state.pose_raw_done = True
                    st.session_state.pose_done = True
                    st.session_state.subject_locked = True
                    st.info("Abre **4 · Resultados 2D**, revisa el desfase/segmento válido y calcula. Después se eliminarán los vídeos temporales.")
            except Exception as e:
                st.error(f"Error durante el análisis: {e}")
                with st.expander("Detalles técnicos"):
                    st.code(traceback.format_exc())

        # Segunda etapa exclusivamente multipersona: elección humana + bloqueo de identidad.
        if manual_mode and st.session_state.get("pose_raw_done") and not st.session_state.get("subject_locked"):
            st.markdown("### Selección manual del sujeto biomecánico")
            st.warning("Selecciona al **paciente**, no al terapeuta/acompañante. La app conservará esa identidad y rechazará frames ambiguos en lugar de cambiar de persona.")

            sel1=st.session_state.get("subject_selection_cam1")
            if sel1:
                st.markdown("#### Cámara 1")
                if st.session_state.get("subject_preview_cam1"):
                    st.image(st.session_state.subject_preview_cam1, caption=f"Frame de selección {sel1['frame']} · sujetos ordenados de izquierda a derecha", use_container_width=True)
                labels1=[c["label"] for c in sel1["candidates"]]
                choice1=st.radio("Paciente en cámara 1", labels1, key="manual_subject_choice_cam1", horizontal=True)
            else:
                choice1=None

            choice2=None
            sel2=st.session_state.get("subject_selection_cam2")
            if current_two and sel2:
                st.markdown("#### Cámara 2")
                if st.session_state.get("subject_preview_cam2"):
                    st.image(st.session_state.subject_preview_cam2, caption=f"Frame de selección {sel2['frame']} · selecciona a la misma persona física", use_container_width=True)
                labels2=[c["label"] for c in sel2["candidates"]]
                choice2=st.radio("Paciente en cámara 2", labels2, key="manual_subject_choice_cam2", horizontal=True)

            if st.button("🔒 Bloquear sujeto y preparar análisis biomecánico", type="primary", use_container_width=True):
                try:
                    c1=next(c for c in sel1["candidates"] if c["label"]==choice1)
                    df1,t1=load_pose_dataframe_tracked(Path(st.session_state.pose_json1), sel1["frame"], c1["person_index"])
                    if df1.empty or len(df1)<10:
                        raise RuntimeError("El seguimiento del sujeto seleccionado en cam01 no produce suficientes frames fiables.")
                    t1["selected_label"] = choice1
                    st.session_state.selected_subject_label_cam1 = choice1
                    st.session_state.analysis_df=df1
                    st.session_state.tracking_info_cam1=t1
                    q1=camera_pose_quality(df1)
                    st.session_state.camera1_quality=q1

                    if current_two:
                        c2=next(c for c in sel2["candidates"] if c["label"]==choice2)
                        df2,t2=load_pose_dataframe_tracked(Path(st.session_state.pose_json2), sel2["frame"], c2["person_index"])
                        if df2.empty or len(df2)<10:
                            raise RuntimeError("El seguimiento del sujeto seleccionado en cam02 no produce suficientes frames fiables.")
                        t2["selected_label"] = choice2
                        st.session_state.selected_subject_label_cam2 = choice2
                        st.session_state.analysis_df2=df2
                        st.session_state.tracking_info_cam2=t2
                        q2=camera_pose_quality(df2)
                        st.session_state.camera2_quality=q2
                        meta1=st.session_state.meta1; meta2=st.session_state.meta2
                        off,corr,sq=estimate_sync_offset(df1,meta1["fps"],df2,meta2["fps"],max_offset_s=2.0)
                        st.session_state.sync_offset_auto_s=off
                        st.session_state.sync_offset_user_s=off
                        st.session_state.sync_correlation=corr
                        st.session_state.sync_quality=sq
                        ready,reasons=readiness_3d(st.session_state.mode,q1,q2,sq,st.session_state.get("selected_calibration_name"))
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            sb_update_session_3d(
                                st.session_state.cloud_session_id,
                                sync_offset_s=off,
                                sync_correlation=corr if np.isfinite(corr) else None,
                                sync_quality=sq,
                                calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                ready_3d=bool(ready),
                            )
                    else:
                        st.session_state.analysis_df2=None

                    st.session_state.subject_locked=True
                    st.session_state.pose_done=True
                    st.success("🔒 Sujeto biomecánico bloqueado. La otra persona queda excluida del análisis.")
                    st.write(
                        f"Continuidad de identidad cam01: **{t1.get('identity_continuity_pct',np.nan):.1f}%** · "
                        f"frames excluidos: **{t1.get('frames_excluded',0)}** · calidad: **{t1.get('quality','')}**."
                    )
                    if current_two:
                        st.write(
                            f"Continuidad de identidad cam02: **{t2.get('identity_continuity_pct',np.nan):.1f}%** · "
                            f"frames excluidos: **{t2.get('frames_excluded',0)}** · calidad: **{t2.get('quality','')}**."
                        )
                    st.info("Abre **4 · Resultados 2D** y calcula el intervalo válido. Las métricas se obtendrán únicamente del sujeto seleccionado.")
                except Exception as e:
                    st.error(f"No se pudo bloquear el sujeto: {e}")
                    with st.expander("Detalles técnicos"):
                        st.code(traceback.format_exc())

        elif manual_mode and st.session_state.get("subject_locked"):
            t1=st.session_state.get("tracking_info_cam1") or {}
            st.success("🔒 Sujeto biomecánico bloqueado.")
            if t1:
                st.caption(
                    f"Cam01 · continuidad identidad {t1.get('identity_continuity_pct',np.nan):.1f}% · "
                    f"excluidos {t1.get('frames_excluded',0)}/{t1.get('frames_total',0)} frames · "
                    f"cambios de identidad evitados {t1.get('switches_prevented',0)}."
                )

with tabs[3]:
    st.subheader("Resultados 2D complementarios")
    df1 = st.session_state.get("analysis_df")
    df2 = st.session_state.get("analysis_df2")
    meta1 = st.session_state.get("meta1")
    meta2 = st.session_state.get("meta2")
    current_two = st.session_state.get("mode", "").startswith("2 cámaras")

    if (df1 is None or not st.session_state.get("pose_done") or not meta1) and not st.session_state.get("metrics_done"):
        st.info("Ejecuta primero **Analizar marcha**.")

    if df1 is not None and meta1:
        if current_two and df2 is not None and meta2:
            st.markdown("### Sincronización de cámaras")
            auto = float(st.session_state.get("sync_offset_auto_s", 0.0))
            corr = st.session_state.get("sync_correlation", np.nan)
            sq = st.session_state.get("sync_quality", "No calculable")
            st.caption("Convención: un valor positivo significa que el mismo evento aparece más tarde en cam02; para alinear, cam02 se avanza/recorta ese tiempo.")
            st.write(f"Estimación automática: **{auto:+.3f} s** · correlación **{fmt(corr,2)}** · calidad **{sq}**")
            max_sync = 2.0
            sync_user = st.number_input(
                "Desfase cam02 respecto a cam01 (s)",
                min_value=-max_sync, max_value=max_sync,
                value=float(st.session_state.get("sync_offset_user_s", auto)),
                step=0.01,
                help="Puedes corregir manualmente el desfase si conoces el evento de sincronización.",
            )
            st.session_state.sync_offset_user_s = float(sync_user)
            dur1, dur2 = float(meta1["duration"]), float(meta2["duration"])
            common_start = max(0.0, -float(sync_user))
            common_end = min(dur1, dur2 - float(sync_user))
            if common_end <= common_start + 1.0:
                st.error("No queda un intervalo temporal común suficiente con el desfase seleccionado.")
            else:
                start_s, end_s = st.slider(
                    "Intervalo válido común (tiempo de cam01)",
                    float(round(common_start,2)), float(round(common_end,2)),
                    (float(round(common_start,2)), float(round(common_end,2))),
                    step=0.02,
                )
                st.caption(f"Cam02 analizará aproximadamente {start_s+sync_user:.2f}–{end_s+sync_user:.2f} s para representar el mismo intervalo físico.")
                if st.button("Calcular ambas vistas, guardar histórico y eliminar vídeos", type="primary", use_container_width=True):
                    try:
                        f1a, f1b = int(round(start_s*meta1["fps"])), int(round(end_s*meta1["fps"]))
                        s2a, s2b = start_s + sync_user, end_s + sync_user
                        f2a, f2b = int(round(s2a*meta2["fps"])), int(round(s2b*meta2["fps"]))
                        m_front, chart_front, seg_front = compute_metrics(
                            df1, float(meta1["fps"]), f1a, f1b, "Frontal/posterior", st.session_state.get("assistive_device","Sin ayuda"), st.session_state.get("scale_cm_per_px",0.0)
                        )
                        m_lat, chart_lat, seg_lat = compute_metrics(
                            df2, float(meta2["fps"]), f2a, f2b, "Lateral", st.session_state.get("assistive_device","Sin ayuda"), 0.0
                        )
                        metrics = prefix_metrics(m_front, "front", "Frontal/posterior") + prefix_metrics(m_lat, "lateral", "Lateral")
                        metrics += tracking_metrics(st.session_state.get("tracking_info_cam1"), "front", "Frontal/posterior")
                        metrics += tracking_metrics(st.session_state.get("tracking_info_cam2"), "lateral", "Lateral")
                        metrics += [
                            {"key":"sync_offset_cam02_s","label":"Desfase cam02 vs cam01","value":sync_user,"unit":"s","quality":sq,"notes":"Sincronización temporal experimental; valor positivo = evento más tardío en cam02."},
                            {"key":"sync_correlation","label":"Correlación de sincronización automática","value":corr,"unit":"r","quality":sq,"notes":"Correlación heurística de movimiento corporal vertical entre cámaras."},
                        ]
                        q1 = st.session_state.get("camera1_quality") or camera_pose_quality(df1)
                        q2 = st.session_state.get("camera2_quality") or camera_pose_quality(df2)
                        ready, reasons = readiness_3d(st.session_state.mode, q1, q2, sq, st.session_state.get("selected_calibration_name"))
                        metrics.append({"key":"ready_3d_flag","label":"Preparación para triangulación 3D","value":1.0 if ready else 0.0,"unit":"bool","quality":"Preparado" if ready else "Pendiente","notes":"; ".join(reasons) if reasons else "Dos cámaras, visibilidad suficiente, sincronización aceptable y perfil de calibración seleccionado."})
                        # v0.10.1 · snapshots ligeros de ambas vistas.
                        snap_front = _make_cycle_snapshot(seg_front, float(meta1["fps"]))
                        snap_lat = _make_cycle_snapshot(seg_lat, float(meta2["fps"]))
                        st.session_state.cycle_seg_front = snap_front["seg"] if snap_front else seg_front.copy()
                        st.session_state.cycle_seg_lateral = snap_lat["seg"] if snap_lat else seg_lat.copy()
                        st.session_state.cycle_fps_front = float(meta1["fps"])
                        st.session_state.cycle_fps_lateral = float(meta2["fps"])
                        st.session_state.cycle_support_left_front = snap_front["left"] if snap_front else None
                        st.session_state.cycle_support_right_front = snap_front["right"] if snap_front else None
                        st.session_state.cycle_support_left_lateral = snap_lat["left"] if snap_lat else None
                        st.session_state.cycle_support_right_lateral = snap_lat["right"] if snap_lat else None
                        # La pestaña 9 usa frontal por defecto porque es la vista clínica activa.
                        st.session_state.cycle_seg = st.session_state.cycle_seg_front
                        st.session_state.cycle_fps = st.session_state.cycle_fps_front
                        st.session_state.cycle_support_left = st.session_state.cycle_support_left_front
                        st.session_state.cycle_support_right = st.session_state.cycle_support_right_front
                        st.session_state.cycle_source_view = "Frontal/posterior"
                        st.session_state.cycle_phase_review_base = None
                        st.session_state.cycle_phase_review_bytes = None
                        st.session_state.cycle_phase_review_speed = None

                        st.session_state.metrics = metrics
                        st.session_state.chart_front = chart_front
                        st.session_state.chart_lateral = chart_lat
                        st.session_state.metrics_done = True
                        st.session_state.ready_3d = ready
                        st.session_state.ready_3d_reasons = reasons
                        cloud_saved = False
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            try:
                                sb_save_metrics(st.session_state.cloud_session_id, metrics, start_s, end_s)
                                sb_update_session_3d(
                                    st.session_state.cloud_session_id,
                                    sync_offset_s=float(sync_user),
                                    sync_correlation=float(corr) if np.isfinite(corr) else None,
                                    sync_quality=sq,
                                    calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                    ready_3d=bool(ready),
                                )
                                cloud_saved = True
                            except Exception as sb_e:
                                st.warning(f"Resultados calculados, pero no se pudieron guardar en Supabase: {sb_e}")
                        session_dir = Path(st.session_state.session_dir)

                        # v0.10.3 · copias limpias efímeras en memoria para auditoría
                        # del sujeto seleccionado en la pestaña 9.
                        try:
                            vp1 = Path(st.session_state.video1_path)
                            if vp1.exists():
                                st.session_state.cycle_video_clean_bytes = vp1.read_bytes()
                        except Exception:
                            st.session_state.cycle_video_clean_bytes = None
                        try:
                            vp2 = Path(st.session_state.video2_path)
                            if vp2.exists():
                                st.session_state.cycle_video2_clean_bytes = vp2.read_bytes()
                        except Exception:
                            st.session_state.cycle_video2_clean_bytes = None

                        try:
                            out1 = session_dir / "gait_front_web.mp4"
                            made1 = render_angle_video(Path(st.session_state.video1_path), df1, out1, "Frontal/posterior", st.session_state.get("assistive_device","Sin ayuda"))
                            if made1 and made1.exists():
                                st.session_state.annotated_video_bytes = made1.read_bytes()
                            out2 = session_dir / "gait_lateral_web.mp4"
                            made2 = render_angle_video(Path(st.session_state.video2_path), df2, out2, "Lateral", st.session_state.get("assistive_device","Sin ayuda"))
                            if made2 and made2.exists():
                                st.session_state.annotated_video2_bytes = made2.read_bytes()
                        except Exception:
                            st.session_state.annotated_video_bytes = None
                            st.session_state.annotated_video2_bytes = None
                        cleanup_temp_session(session_dir)
                        st.session_state.temp_deleted = True
                        st.session_state.analysis_df = None
                        st.session_state.analysis_df2 = None
                        st.success("✅ Resultados calculados y mostrados. ✅ Vídeos y archivos Pose2Sim eliminados del servidor temporal." + (" ✅ Histórico guardado en Supabase." if cloud_saved else " ⚠️ Histórico no guardado en Supabase."))
                    except Exception as e:
                        st.error(str(e))
                        with st.expander("Detalles técnicos"):
                            st.code(traceback.format_exc())
        else:
            fps = float(meta1["fps"])
            duration = (int(df1.frame.max()) + 1) / fps
            start_s, end_s = st.slider(
                "Intervalo válido (segundos)", 0.0, float(round(duration,2)),
                (0.0,float(round(duration,2))), step=max(0.01, round(1/fps,2))
            )
            if st.button("Calcular, guardar histórico y eliminar vídeo", type="primary", use_container_width=True):
                try:
                    start_frame = int(round(start_s*fps)); end_frame = int(round(end_s*fps))
                    metrics, chart, seg = compute_metrics(df1, fps, start_frame, end_frame, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"), st.session_state.get("scale_cm_per_px",0.0))
                    metrics += tracking_metrics(st.session_state.get("tracking_info_cam1"), "", st.session_state.get("view",""))

                    # v0.10.1 · preservar el segmento YA usado para los resultados
                    # antes de eliminar vídeo/JSON temporales.
                    snap = _make_cycle_snapshot(seg, fps)
                    st.session_state.cycle_seg = snap["seg"] if snap else seg.copy()
                    st.session_state.cycle_fps = float(fps)
                    st.session_state.cycle_support_left = snap["left"] if snap else None
                    st.session_state.cycle_support_right = snap["right"] if snap else None
                    st.session_state.cycle_source_view = st.session_state.get("view","")
                    st.session_state.cycle_phase_review_base = None
                    st.session_state.cycle_phase_review_bytes = None
                    st.session_state.cycle_phase_review_speed = None

                    st.session_state.metrics = metrics
                    st.session_state.chart = chart
                    st.session_state.metrics_done = True
                    cloud_saved = False
                    if sb_ready() and st.session_state.get("cloud_session_id"):
                        try:
                            sb_save_metrics(st.session_state.cloud_session_id, metrics, start_s, end_s)
                            cloud_saved = True
                        except Exception as sb_e:
                            st.warning(f"Resultados calculados, pero no se pudieron guardar en Supabase: {sb_e}")
                    session_dir = Path(st.session_state.session_dir)

                    # v0.10.3 · conservar en memoria SOLO durante la sesión una copia
                    # limpia del vídeo fuente. No se escribe en Supabase.
                    try:
                        vp_clean = Path(st.session_state.video1_path)
                        if vp_clean.exists():
                            st.session_state.cycle_video_clean_bytes = vp_clean.read_bytes()
                    except Exception:
                        st.session_state.cycle_video_clean_bytes = None

                    try:
                        out = session_dir / "gait_angles_web.mp4"
                        made = render_angle_video(Path(st.session_state.video1_path), df1, out, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"))
                        if made and made.exists():
                            st.session_state.annotated_video_bytes = made.read_bytes()
                    except Exception:
                        st.session_state.annotated_video_bytes = None
                    cleanup_temp_session(session_dir)
                    st.session_state.temp_deleted = True
                    st.session_state.analysis_df = None
                    st.success("✅ Resultados calculados y mostrados. ✅ Vídeo y archivos Pose2Sim eliminados del servidor temporal." + (" ✅ Histórico guardado en Supabase." if cloud_saved else " ⚠️ Histórico no guardado en Supabase."))
                except Exception as e:
                    st.error(str(e))
                    with st.expander("Detalles técnicos"):
                        st.code(traceback.format_exc())

    if st.session_state.get("metrics_done"):
        metrics = st.session_state.metrics
        st.markdown("### Resumen")
        if st.session_state.get("mode", "").startswith("2 cámaras"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Cadencia lateral", fmt(metric_value(metrics,"lateral_cadence_exp"),1)+" pasos/min")
            c2.metric("Tracking frontal", fmt(metric_value(metrics,"front_good_frames_pct"),1)+" %")
            c3.metric("Tracking lateral", fmt(metric_value(metrics,"lateral_good_frames_pct"),1)+" %")
            c4.metric("Desfase cam02", fmt(metric_value(metrics,"sync_offset_cam02_s"),2)+" s")
            st.caption(f"Cadencia lateral ({reference_text_for_metric('lateral_cadence_exp')})")
            sl=metric_value(metrics,"lateral_stance_time_l_2d"); sr=metric_value(metrics,"lateral_stance_time_r_2d"); sa=metric_value(metrics,"lateral_stance_asymmetry_2d")
            qsl=metric_value(metrics,"lateral_support_segmentation_score_l"); qsr=metric_value(metrics,"lateral_support_segmentation_score_r")
            if sl is not None and sr is not None and np.isfinite(sl) and np.isfinite(sr):
                side="IZQUIERDA" if sl>sr else ("DERECHA" if sr>sl else "SIMILAR")
                st.info(f"Apoyo 2D por ciclos continuos (lateral): I {sl:.2f} s · D {sr:.2f} s · asimetría {fmt(sa,1)} % · mayor tiempo estimado: **{side}**. ({reference_text_for_metric('lateral_stance_asymmetry_2d')})")
            else:
                st.warning(f"Tiempo de apoyo lateral NO informado: la segmentación apoyo/oscilación no superó el control temporal (I {fmt(qsl,0)}/100 · D {fmt(qsr,0)}/100).")
            st.markdown("#### Consistencia temporal · cámara lateral")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Eventos detectados", fmt(metric_value(metrics,"lateral_step_events_detected"),0))
            t2.metric("Duración segmento", fmt(metric_value(metrics,"lateral_segment_duration_s"),2)+" s")
            t3.metric("Eventos esperados", fmt(metric_value(metrics,"lateral_expected_steps_from_cadence"),1))
            t4.metric("Discrepancia", fmt(metric_value(metrics,"lateral_step_count_consistency_error_pct"),1)+" %")
            cq = metric_quality(metrics,"lateral_step_count_consistency_error_pct") or "No calculable"
            if cq == "Alta": st.success(f"Consistencia interna: {cq}")
            elif cq == "Moderada": st.info(f"Consistencia interna: {cq}")
            else: st.warning(f"Consistencia interna: {cq}. Conviene revisar visualmente los eventos detectados.")
            st.caption("Control interno: número de eventos de alternancia detectados ↔ cadencia estimada ↔ duración. Los eventos siguen siendo experimentales y no equivalen todavía a heel-strikes validados.")
            st.markdown("### Resumen biomecánico")
            for paragraph in biomech_summary(metrics, "Lateral", prefix="lateral_"):
                st.write(paragraph)
            for paragraph in biomech_summary(metrics, "Frontal/posterior", prefix="front_")[1:]:
                st.write(paragraph)
            ready = bool(st.session_state.get("ready_3d", False))
            if ready:
                st.success("🧭 Sesión preparada para una futura triangulación 3D: dos poses, sincronización aceptable, visibilidad suficiente y calibración seleccionada.")
            else:
                st.warning("La sesión todavía no cumple todos los requisitos de preparación 3D.")
                for reason in st.session_state.get("ready_3d_reasons", []):
                    st.write(f"• {reason}")
            st.warning("La v0.7 **no triangula todavía coordenadas 3D**. Los resultados siguientes continúan siendo 2D, pero cada plano se interpreta desde la cámara apropiada.")
            chart_front = st.session_state.get("chart_front")
            chart_lat = st.session_state.get("chart_lateral")
            if chart_front is not None:
                st.markdown("### Cámara 1 · frontal/posterior")
                for title, cols in [
                    ("Pelvis y tronco", ["Oblicuidad pélvica","Inclinación lateral del tronco","Relación hombros-pelvis"]),
                    ("Hombros", ["Oblicuidad de hombros"]),
                    ("Cadera / rodilla · eje frontal", ["Rodilla frontal izquierda","Rodilla frontal derecha"]),
                    ("Pie · orientación distal", ["Orientación pie izquierda","Orientación pie derecha"]),
                    ("Pie · inclinación del retropié", ["Retropié izquierda","Retropié derecha"]),
                    ("Avanzado frontal", ["CoM proxy lateral (px)","Valgo dinámico proyectado I","Valgo dinámico proyectado D"]),
                ]:
                    with st.expander(title, expanded=(title == "Pelvis y tronco")):
                        st.line_chart(chart_front.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo proyectado (°)")
            if chart_lat is not None:
                st.markdown("### Cámara 2 · lateral")
                for title, cols in [
                    ("Rodillas", ["Rodilla izquierda","Rodilla derecha"]),
                    ("Caderas", ["Cadera izquierda","Cadera derecha"]),
                    ("Tobillos / pie", ["Tobillo izquierda","Tobillo derecha"]),
                    ("Hombros", ["Hombro izquierda","Hombro derecha"]),
                ]:
                    with st.expander(title, expanded=(title == "Rodillas")):
                        st.line_chart(chart_lat.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo 2D (°)")
            if st.session_state.get("annotated_video_bytes") or st.session_state.get("annotated_video2_bytes"):
                st.markdown("### Vídeos con esqueleto + métricas")
                v1,v2 = st.columns(2)
                with v1:
                    if st.session_state.get("annotated_video_bytes"):
                        st.caption("Cámara 1 · frontal/posterior")
                        st.video(st.session_state.annotated_video_bytes)
                with v2:
                    if st.session_state.get("annotated_video2_bytes"):
                        st.caption("Cámara 2 · lateral")
                        st.video(st.session_state.annotated_video2_bytes)
        else:
            chart = st.session_state.chart
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Cadencia estimada", fmt(metric_value(metrics,"cadence_exp"),1)+" pasos/min")
            c2.metric("Regularidad temporal", fmt(metric_value(metrics,"regularity_cv"),1)+" % CV")
            c3.metric("Tracking válido", fmt(metric_value(metrics,"good_frames_pct"),1)+" %")
            c4.metric("Asimetría temporal", fmt(metric_value(metrics,"temporal_asymmetry_exp"),1)+" %")
            st.caption(f"Cadencia ({reference_text_for_metric('cadence_exp')}) · Variabilidad ({reference_text_for_metric('regularity_cv')}) · Asimetría ({reference_text_for_metric('temporal_asymmetry_exp')})")
            sl=metric_value(metrics,"stance_time_l_2d"); sr=metric_value(metrics,"stance_time_r_2d"); sa=metric_value(metrics,"stance_asymmetry_2d")
            qsl=metric_value(metrics,"support_segmentation_score_l"); qsr=metric_value(metrics,"support_segmentation_score_r")
            if sl is not None and sr is not None and np.isfinite(sl) and np.isfinite(sr):
                side="IZQUIERDA" if sl>sr else ("DERECHA" if sr>sl else "SIMILAR")
                st.info(f"Apoyo 2D por ciclos continuos: I {sl:.2f} s · D {sr:.2f} s · asimetría {fmt(sa,1)} % · mayor tiempo estimado: **{side}**. ({reference_text_for_metric('stance_asymmetry_2d')})")
            else:
                st.warning(f"Tiempo de apoyo NO informado: la segmentación apoyo/oscilación no superó el control temporal en ambos lados (I {fmt(qsl,0)}/100 · D {fmt(qsr,0)}/100). Se evita mostrar una duración potencialmente falsa.")
            st.markdown("#### Consistencia número de pasos · cadencia · duración")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Eventos detectados", fmt(metric_value(metrics,"step_events_detected"),0))
            t2.metric("Duración segmento", fmt(metric_value(metrics,"segment_duration_s"),2)+" s")
            t3.metric("Eventos esperados", fmt(metric_value(metrics,"expected_steps_from_cadence"),1))
            t4.metric("Discrepancia", fmt(metric_value(metrics,"step_count_consistency_error_pct"),1)+" %")
            cq = metric_quality(metrics,"step_count_consistency_error_pct") or "No calculable"
            if cq == "Alta": st.success(f"Consistencia interna: {cq}")
            elif cq == "Moderada": st.info(f"Consistencia interna: {cq}")
            else: st.warning(f"Consistencia interna: {cq}. Conviene revisar visualmente los eventos detectados.")
            st.caption("Los eventos detectados son alternancias distales experimentales. La comprobación sirve para verificar coherencia interna, no para convertirlos automáticamente en heel-strikes clínicamente validados.")
            st.markdown("### Resumen biomecánico")
            for paragraph in biomech_summary(metrics, st.session_state.get("view", "")):
                st.write(paragraph)
            if "Frontal" in st.session_state.get("view", ""):
                st.markdown("### Biomecánica frontal/posterior 2D proyectada")
                st.warning("La rotación axial de cadera y la pronación son movimientos 3D. Aquí se muestran proxies 2D y no deben interpretarse como diagnóstico aislado.")
                for title, cols in [
                    ("Pelvis y tronco", ["Oblicuidad pélvica","Inclinación lateral del tronco","Relación hombros-pelvis"]),
                    ("Hombros", ["Oblicuidad de hombros"]),
                    ("Cadera / rodilla · eje frontal", ["Rodilla frontal izquierda","Rodilla frontal derecha"]),
                    ("Pie · orientación distal", ["Orientación pie izquierda","Orientación pie derecha"]),
                    ("Pie · inclinación del retropié", ["Retropié izquierda","Retropié derecha"]),
                    ("Avanzado frontal", ["CoM proxy lateral (px)","Valgo dinámico proyectado I","Valgo dinámico proyectado D"]),
                ]:
                    with st.expander(title, expanded=(title == "Pelvis y tronco")):
                        st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo proyectado (°)")
            else:
                st.markdown("### Cinemática sagital 2D proyectada")
                for title, cols in [
                    ("Rodillas", ["Rodilla izquierda","Rodilla derecha"]),
                    ("Caderas", ["Cadera izquierda","Cadera derecha"]),
                    ("Tobillos / pie", ["Tobillo izquierda","Tobillo derecha"]),
                    ("Hombros", ["Hombro izquierda","Hombro derecha"]),
                ]:
                    with st.expander(title, expanded=(title == "Rodillas")):
                        st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo 2D (°)")
            if st.session_state.get("annotated_video_bytes"):
                st.markdown("### Vídeo con esqueleto + ángulos")
                st.video(st.session_state.annotated_video_bytes)
        with st.expander("Todas las métricas y calidad"):
            _mdf=pd.DataFrame(metrics).copy()
            _mdf["reference"]=_mdf["key"].map(reference_text_for_metric)
            st.dataframe(_mdf[["label","value","unit","quality","reference","notes"]], use_container_width=True, hide_index=True)
        st.caption("Cadencia, alternancia y sincronización automática permanecen experimentales hasta validación específica del protocolo.")

with tabs[4]:
    st.subheader("Pacientes / Evolución longitudinal")
    st.caption("Paciente → registros → comparación → evolución. Los vídeos no se almacenan; el histórico usa sesiones y métricas guardadas en Supabase.")
    if not sb_ready():
        st.info("Configura Supabase para activar el histórico persistente.")
    else:
        try:
            pats = sb_list_patients()
            if pats.empty:
                st.info("Todavía no hay pacientes en el histórico.")
            else:
                codes = pats.code.tolist()
                default = codes.index(patient) if patient in codes else 0
                selected = st.selectbox("Paciente / código", codes, index=default, key="history_patient")
                hist_all = sb_patient_history(selected)
                if hist_all.empty:
                    st.info("Este paciente todavía no tiene métricas guardadas.")
                else:
                    hist_all["created_dt"] = pd.to_datetime(hist_all["created_at"], utc=True)
                    hist_all["fecha"] = hist_all["created_dt"].dt.strftime("%d/%m/%Y %H:%M")
                    sessions = hist_all[[c for c in ["session_id","created_dt","fecha","record_name","mode","view","assistive_device","frontal_orientation","duration_s","duration_cam2_s","segment_start_s","segment_end_s","analysis_status","ready_3d"] if c in hist_all.columns]].drop_duplicates("session_id").sort_values("created_dt")

                    a,b,c,d = st.columns(4)
                    a.metric("Registros", sessions.session_id.nunique())
                    a0=sessions.created_dt.min(); a1=sessions.created_dt.max()
                    b.metric("Primer registro", a0.strftime("%d/%m/%Y") if pd.notna(a0) else "—")
                    c.metric("Último registro", a1.strftime("%d/%m/%Y") if pd.notna(a1) else "—")
                    d.metric("Seguimiento", f"{max(0,(a1-a0).days)} días" if pd.notna(a0) and pd.notna(a1) else "—")

                    st.markdown("### Línea de tiempo de registros")
                    timeline=sessions.copy()
                    timeline["Registro"] = timeline["record_name"].fillna("Marcha")
                    timeline["Fecha"] = timeline["fecha"]
                    timeline["Vista"] = timeline["view"].fillna("")
                    timeline["Ayuda"] = timeline["assistive_device"].fillna("Sin ayuda")
                    timeline["Estado"] = timeline["analysis_status"].fillna("")
                    st.dataframe(timeline[["Fecha","Registro","Vista","Ayuda","Estado"]], use_container_width=True, hide_index=True)

                    with st.expander("🗑️ Gestionar / borrar registros duplicados", expanded=False):
                        st.caption(
                            "Selecciona uno o varios registros para eliminarlos del histórico. "
                            "Se borrarán la sesión y sus métricas de Supabase. El paciente no se elimina."
                        )
                        delete_labels = {
                            r.session_id: (
                                f"{r.fecha} · {r.record_name or 'Marcha'} · "
                                f"{r.view or 'Vista no especificada'} · "
                                f"{r.assistive_device or 'Sin ayuda'}"
                            )
                            for _, r in sessions.sort_values("created_dt", ascending=False).iterrows()
                        }
                        delete_ids = st.multiselect(
                            "Registros a borrar",
                            options=list(delete_labels.keys()),
                            format_func=lambda x: delete_labels[x],
                            key=f"delete_sessions_{selected}",
                            placeholder="Selecciona los registros duplicados o no válidos",
                        )
                        if delete_ids:
                            st.warning(
                                f"Vas a borrar permanentemente {len(delete_ids)} registro(s) "
                                "y todas sus métricas asociadas. Esta acción no se puede deshacer."
                            )
                            confirm_delete = st.checkbox(
                                "Confirmo que quiero borrar permanentemente los registros seleccionados",
                                key=f"confirm_delete_{selected}",
                            )
                            if st.button(
                                f"Eliminar {len(delete_ids)} registro(s)",
                                type="primary",
                                disabled=not confirm_delete,
                                key=f"delete_button_{selected}",
                            ):
                                deleted = 0
                                errors = []
                                for sid in delete_ids:
                                    try:
                                        sb_delete_session(sid)
                                        deleted += 1
                                    except Exception as exc:
                                        errors.append(f"{delete_labels.get(sid, sid)}: {exc}")
                                if deleted:
                                    st.success(
                                        f"Se han eliminado {deleted} registro(s) y sus métricas asociadas."
                                    )
                                if errors:
                                    st.error("No se pudieron borrar todos los registros:\n\n" + "\n".join(errors))
                                if deleted and not errors:
                                    st.rerun()

                    st.markdown("### Evolución de todos los registros")
                    f1,f2,f3 = st.columns(3)
                    aid_options=["Todas"]+sorted([x for x in hist_all.assistive_device.dropna().unique().tolist()]) if "assistive_device" in hist_all.columns else ["Todas"]
                    aid_filter=f1.selectbox("Ayuda técnica", aid_options, key="hist_aid")
                    view_options=["Todas"]+sorted([x for x in hist_all.view.dropna().unique().tolist()]) if "view" in hist_all.columns else ["Todas"]
                    view_filter=f2.selectbox("Vista", view_options, key="hist_view")
                    ref_mode=f3.selectbox("Referencia", ["Poblacional publicada", "Primer registro del paciente", "Sin referencia"], key="hist_ref")
                    hist=hist_all.copy()
                    if aid_filter!="Todas": hist=hist[hist.assistive_device==aid_filter]
                    if view_filter!="Todas": hist=hist[hist.view==view_filter]
                    labels = hist[["metric_key","metric_label","unit"]].drop_duplicates().sort_values("metric_label")
                    if labels.empty:
                        st.info("No hay métricas con esos filtros.")
                    else:
                        label = st.selectbox("Parámetro", labels.metric_label.tolist(), key="hist_metric")
                        key = labels.loc[labels.metric_label==label,"metric_key"].iloc[0]
                        h = hist[hist.metric_key==key].copy().sort_values("created_dt")
                        unit = h.unit.iloc[0] if len(h) else ""
                        if not h.empty:
                            plot=h[["fecha","value"]].dropna().set_index("fecha").rename(columns={"value":label})
                            st.line_chart(plot, x_label="Registro", y_label=f"{label} ({unit})" if unit else label)
                            first=float(h.value.dropna().iloc[0]) if h.value.notna().any() else np.nan
                            last=float(h.value.dropna().iloc[-1]) if h.value.notna().any() else np.nan
                            prev=float(h.value.dropna().iloc[-2]) if h.value.notna().sum()>=2 else np.nan
                            q1,q2,q3,q4=st.columns(4)
                            q1.metric("Basal", f"{first:.2f} {unit}" if np.isfinite(first) else "—")
                            q2.metric("Último", f"{last:.2f} {unit}" if np.isfinite(last) else "—", delta=f"{last-first:+.2f}" if np.isfinite(first) and np.isfinite(last) else None)
                            q3.metric("Δ vs basal", f"{((last-first)/abs(first)*100):+.1f} %" if np.isfinite(first) and first!=0 and np.isfinite(last) else "—")
                            q4.metric("Δ vs anterior", f"{last-prev:+.2f} {unit}" if np.isfinite(prev) and np.isfinite(last) else "—")
                            if ref_mode=="Poblacional publicada":
                                ref=reference_for_metric(key)
                                if ref:
                                    st.info(f"**Referencia contextual:** {ref['low']:.2f}–{ref['high']:.2f} {ref['unit']} · {reference_position(last,ref)}. {ref['population']}. {ref['applicability']}")
                                    st.caption(f"Fuente: {ref['source']} · DOI {ref['doi']} · Biblioteca PhysioSentinel {REFERENCE_LIBRARY_VERSION}")
                                else:
                                    st.warning("Esta métrica no tiene todavía una referencia poblacional suficientemente compatible con el método de PhysioSentinel. Se prioriza la evolución intraindividual.")
                            elif ref_mode=="Primer registro del paciente":
                                st.info("El primer registro filtrado se utiliza como referencia individual. Esto describe cambio respecto al basal y no implica por sí mismo mejoría o empeoramiento.")

                    st.markdown("### Comparar dos registros")
                    sess_labels={r.session_id:f"{r.fecha} · {r.record_name or 'Marcha'} · {r.view or ''} · {r.assistive_device or 'Sin ayuda'}" for _,r in sessions.iterrows()}
                    ids=list(sess_labels.keys())
                    if len(ids)<2:
                        st.info("Se necesitan al menos dos registros para una comparación directa.")
                    else:
                        ca,cb=st.columns(2)
                        sid_a=ca.selectbox("Registro A",ids,index=0,format_func=lambda x:sess_labels[x],key="cmp_a")
                        sid_b=cb.selectbox("Registro B",ids,index=len(ids)-1,format_func=lambda x:sess_labels[x],key="cmp_b")
                        if sid_a==sid_b:
                            st.warning("Selecciona dos registros diferentes.")
                        else:
                            ma=hist_all[hist_all.session_id==sid_a][["metric_key","metric_label","value","unit","quality"]].copy()
                            mb=hist_all[hist_all.session_id==sid_b][["metric_key","metric_label","value","unit","quality"]].copy()
                            cmp=ma.merge(mb,on="metric_key",suffixes=("_A","_B"))
                            cmp["Métrica"]=cmp.metric_label_A
                            cmp["A"]=cmp.value_A
                            cmp["B"]=cmp.value_B
                            cmp["Δ"]=cmp["B"]-cmp["A"]
                            cmp["Δ %"]=np.where(cmp["A"].abs()>1e-12,cmp["Δ"]/cmp["A"].abs()*100,np.nan)
                            cmp["Unidad"]=cmp.unit_A
                            st.dataframe(cmp[["Métrica","A","B","Δ","Δ %","Unidad"]].sort_values("Métrica"),use_container_width=True,hide_index=True)
                            sa=sessions[sessions.session_id==sid_a].iloc[0]; sb=sessions[sessions.session_id==sid_b].iloc[0]
                            comparable=(str(sa.get("view",""))==str(sb.get("view","")) and str(sa.get("assistive_device",""))==str(sb.get("assistive_device","")))
                            if comparable: st.success("Condiciones básicas comparables: misma vista y misma ayuda técnica.")
                            else: st.warning("Comparación contextual: cambia la vista y/o la ayuda técnica entre registros. Interpreta los Δ con cautela.")

                    st.markdown("### Resumen longitudinal")
                    st.write(f"**{selected}** dispone de **{sessions.session_id.nunique()} registros** entre {a0.strftime('%d/%m/%Y')} y {a1.strftime('%d/%m/%Y')}. La interpretación prioriza cambios intraindividuales y separa el valor medido de su significado clínico.")
                    st.caption("Un aumento o descenso no se etiqueta automáticamente como mejoría/empeoramiento. La dirección favorable depende de la variable, objetivo terapéutico, velocidad, ayuda técnica y contexto clínico.")
                    with st.expander("Biblioteca de referencias científicas"):
                        st.dataframe(pd.DataFrame(REFERENCE_SOURCES),use_container_width=True,hide_index=True)
                        st.caption("Las bandas solo se muestran cuando existe compatibilidad razonable con la métrica. No se trasladan rangos 3D a proxies 2D.")
        except Exception as e:
            st.error(f"No se pudo leer el histórico: {e}")

with tabs[5]:
    st.subheader("Redacción informe")
    st.caption("Informe clínico estructurado en 3 bloques + síntesis: espaciotemporales, cinemática frontal 2D, coordinación tronco-pelvis e impresión biomecánica. Cada cifra incluye referencia o precaución metodológica.")
    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero el análisis en la pestaña 3 y los resultados en la pestaña 4.")
    else:
        metrics=st.session_state.get("metrics",[])
        tech,patient_txt=generate_reports(
            metrics,
            st.session_state.get("view",view),
            st.session_state.get("patient_code", patient),
            st.session_state.get("record_name", record),
            st.session_state.get("assistive_device", assistive_device),
            patient_age=st.session_state.get("patient_age", 0),
            patient_sex=st.session_state.get("patient_sex", "No especificado"),
            record_date=st.session_state.get("record_date"),
        )
        st.markdown("### Informe de análisis biomecánico de la marcha (2D)")
        tech_edit=st.text_area("Texto técnico editable", value=tech, height=360, key="report_technical")
        st.markdown("### Versión simplificada para el paciente (complementaria)")
        pat_edit=st.text_area("Texto para paciente editable", value=patient_txt, height=320, key="report_patient")
        st.info("Convenio frontal documentado para pelvis: positivo (+) = lado izquierdo elevado / lado derecho descendido; negativo (-) = lado izquierdo descendido / lado derecho elevado. Para otras variables direccionales se evita atribuir signo anatómico cuando la orientación de cámara no permite hacerlo con seguridad.")
        full=f"PHYSIOSENTINEL GAIT {APP_VERSION}\n\nRESUMEN TÉCNICO-CLÍNICO\n{tech_edit}\n\nINFORME PARA EL PACIENTE\n{pat_edit}\n"
        st.download_button("Descargar informe TXT", data=full.encode("utf-8"), file_name=f"PhysioSentinel_Gait_{patient}_{record}.txt", mime="text/plain")

with tabs[6]:
    st.subheader("3D / Calibración")
    st.write("La v0.7 prepara el flujo para **triangulación 3D Pose2Sim** sin presentar todavía resultados 3D como si estuvieran calculados.")
    st.markdown("### 1. Perfil de calibración")
    st.caption("Carga aquí un archivo TOML de calibración de las dos cámaras generado/validado para tu montaje. Se guarda como texto pequeño en Supabase; los vídeos siguen sin persistir.")
    cal_name = st.text_input("Nombre del perfil", value="Consulta_2cam")
    cal_notes = st.text_input("Notas", value="Frontal/posterior + lateral; cámaras fijas")
    cal_file = st.file_uploader("Archivo de calibración TOML", type=["toml"], key="calibration_toml")
    if cal_file is not None:
        try:
            content = cal_file.getvalue().decode("utf-8")
            toml.loads(content)
            st.success("El archivo tiene sintaxis TOML válida. Esto no sustituye la validación geométrica de la calibración.")
            if st.button("Guardar / actualizar perfil en Supabase", type="primary"):
                if not sb_ready():
                    st.error("Supabase no está configurado.")
                elif not cal_name.strip():
                    st.error("Escribe un nombre para el perfil.")
                else:
                    row = sb_upsert_calibration(cal_name.strip(), content, cal_notes, camera_count=2)
                    st.session_state.selected_calibration_name = cal_name.strip()
                    if st.session_state.get("cloud_session_id"):
                        sb_update_session_3d(st.session_state.cloud_session_id, calibration_profile_name=cal_name.strip())
                    st.success(f"Perfil **{cal_name.strip()}** guardado. Ya puede seleccionarse para sesiones de 2 cámaras.")
        except Exception as e:
            st.error(f"El archivo no es TOML válido: {e}")

    st.markdown("### 2. Perfiles disponibles")
    if sb_ready():
        try:
            rows = sb_list_calibrations()
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Todavía no hay perfiles de calibración guardados.")
        except Exception as e:
            st.warning(f"No se pudieron leer perfiles: {e}")

    st.markdown("### 3. Estado de preparación 3D de la sesión actual")
    q1 = st.session_state.get("camera1_quality")
    q2 = st.session_state.get("camera2_quality")
    sq = st.session_state.get("sync_quality", "No calculable")
    ready, reasons = readiness_3d(st.session_state.get("mode",""), q1, q2, sq, st.session_state.get("selected_calibration_name"))
    if ready:
        st.success("La sesión cumple los criterios de preparación definidos en v0.7 para pasar a una futura triangulación 3D.")
    else:
        st.warning("Preparación 3D incompleta.")
        for reason in reasons:
            st.write(f"• {reason}")
    st.info("Siguiente etapa futura: materializar el perfil de calibración en /tmp, aplicar sincronización, ejecutar triangulación Pose2Sim, filtrar puntos 3D y posteriormente calcular cinemática/OpenSim. Esta v0.7 todavía no ejecuta esa etapa.")

with tabs[7]:
    st.subheader("Exportar / Descargar")
    st.caption(
        "Los gráficos, resultados y vídeos anotados se generan para descarga bajo demanda. "
        "No se guardan de forma permanente en Supabase, evitando acumular almacenamiento."
    )

    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero los resultados de la sesión.")
    else:
        metrics = st.session_state.get("metrics", [])
        chart_single = st.session_state.get("chart")
        chart_front = st.session_state.get("chart_front")
        chart_lateral = st.session_state.get("chart_lateral")

        tech, patient_txt = generate_reports(
            metrics,
            st.session_state.get("view", view),
            st.session_state.get("patient_code", patient),
            st.session_state.get("record_name", record),
            st.session_state.get("assistive_device", assistive_device),
            patient_age=st.session_state.get("patient_age", 0),
            patient_sex=st.session_state.get("patient_sex", "No especificado"),
            record_date=st.session_state.get("record_date"),
        )

        # Si el usuario ha editado manualmente los textos del informe, exportar esos.
        tech_export = st.session_state.get("report_technical", tech)
        patient_export = st.session_state.get("report_patient", patient_txt)

        st.markdown("### Contenido disponible")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Métricas", len(metrics))
        graph_cols = 0
        for cdf in (chart_single, chart_front, chart_lateral):
            if isinstance(cdf, pd.DataFrame):
                graph_cols += len([c for c in cdf.columns if c not in ("frame", "time_s")])
        c2.metric("Gráficos", graph_cols)
        c3.metric(
            "Vídeo anotado",
            "Sí" if st.session_state.get("annotated_video_bytes") else "No"
        )
        c4.metric(
            "2ª cámara",
            "Sí" if st.session_state.get("annotated_video2_bytes") else "No"
        )

        st.markdown("### Descargas individuales")

        metrics_df_export = _metrics_dataframe(metrics)
        if not metrics_df_export.empty:
            st.download_button(
                "⬇️ Resultados CSV",
                data=metrics_df_export.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"resultados_{_safe_filename(st.session_state.get('patient_code','paciente'))}_{_safe_filename(st.session_state.get('record_name','marcha'))}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        report_full = (
            f"PHYSIOSENTINEL GAIT {APP_VERSION}\n\n"
            f"INFORME TÉCNICO\n{tech_export}\n\n"
            f"INFORME PARA EL PACIENTE\n{patient_export}\n"
        )
        st.download_button(
            "⬇️ Informes TXT",
            data=report_full.encode("utf-8"),
            file_name=f"informe_{_safe_filename(st.session_state.get('patient_code','paciente'))}_{_safe_filename(st.session_state.get('record_name','marcha'))}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if st.session_state.get("annotated_video_bytes"):
            st.download_button(
                "⬇️ Vídeo anotado cámara 1",
                data=st.session_state.annotated_video_bytes,
                file_name=f"video_anotado_cam01_{_safe_filename(st.session_state.get('patient_code','paciente'))}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        if st.session_state.get("annotated_video2_bytes"):
            st.download_button(
                "⬇️ Vídeo anotado cámara 2",
                data=st.session_state.annotated_video2_bytes,
                file_name=f"video_anotado_cam02_{_safe_filename(st.session_state.get('patient_code','paciente'))}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        st.markdown("### Paquete completo")
        st.write(
            "Incluye resultados CSV/JSON, informes, datos fuente de cada gráfico, "
            "todos los gráficos en PNG y los vídeos anotados disponibles."
        )

        try:
            export_zip = build_export_zip(
                metrics=metrics,
                chart=chart_single,
                chart_front=chart_front,
                chart_lateral=chart_lateral,
                technical_report=tech_export,
                patient_report=patient_export,
                annotated_video_bytes=st.session_state.get("annotated_video_bytes"),
                annotated_video2_bytes=st.session_state.get("annotated_video2_bytes"),
                patient_code=st.session_state.get("patient_code", patient),
                record_name=st.session_state.get("record_name", record),
                patient_age=st.session_state.get("patient_age", 0),
                patient_sex=st.session_state.get("patient_sex", "No especificado"),
                record_date=st.session_state.get("record_date"),
                app_version=APP_VERSION,
            )

            st.download_button(
                "📦 Descargar TODO en ZIP",
                data=export_zip,
                file_name=(
                    f"PhysioSentinel_exportacion_"
                    f"{_safe_filename(st.session_state.get('patient_code','paciente'))}_"
                    f"{_safe_filename(st.session_state.get('record_name','marcha'))}.zip"
                ),
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
            st.success(
                "Paquete preparado en memoria. Al descargarlo no se añade contenido pesado a la base de datos."
            )
        except Exception as e:
            st.error(f"No se pudo generar el paquete de exportación: {e}")
            with st.expander("Detalles técnicos"):
                st.code(traceback.format_exc())


with tabs[8]:
    st.subheader("Analizador interactivo del ciclo de marcha")
    st.caption(
        "Vídeo + tracking + curvas cinemáticas + fases por extremidad sincronizadas. "
        "v0.10.9 añade QC IC/TO y fallback distal bilateral cuando la cadena original falla."
    )

    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero los resultados en la pestaña 4.")
    else:
        # Selección de snapshot. En 2 cámaras puede alternarse entre frontal y lateral.
        two_cycle_views = (
            _valid_dataframe(st.session_state.get("cycle_seg_front"))
            and _valid_dataframe(st.session_state.get("cycle_seg_lateral"))
        )
        cycle_view_choice = "Frontal/posterior"
        if two_cycle_views:
            cycle_view_choice = st.radio(
                "Vista para revisar el ciclo",
                ["Frontal/posterior", "Lateral"],
                horizontal=True,
                key="cycle_review_view",
            )

        if cycle_view_choice == "Lateral" and two_cycle_views:
            seg_cycle = st.session_state.get("cycle_seg_lateral")
            _fps_lat = st.session_state.get("cycle_fps_lateral")
            fps_cycle = float(_fps_lat if _fps_lat is not None else 30.0)
            Lsum = st.session_state.get("cycle_support_left_lateral")
            Rsum = st.session_state.get("cycle_support_right_lateral")
            clean_bytes_cycle = st.session_state.get("cycle_video2_clean_bytes")
            annotated_bytes_cycle = st.session_state.get("annotated_video2_bytes")
            selected_subject_cycle = st.session_state.get("selected_subject_label_cam2")
        else:
            seg_cycle = st.session_state.get("cycle_seg")
            if seg_cycle is None:
                seg_cycle = st.session_state.get("cycle_seg_front")

            fps_cycle = st.session_state.get("cycle_fps")
            if fps_cycle is None:
                fps_cycle = st.session_state.get("cycle_fps_front")
            fps_cycle = float(fps_cycle if fps_cycle is not None else 30.0)

            Lsum = st.session_state.get("cycle_support_left")
            if Lsum is None:
                Lsum = st.session_state.get("cycle_support_left_front")

            Rsum = st.session_state.get("cycle_support_right")
            if Rsum is None:
                Rsum = st.session_state.get("cycle_support_right_front")

            clean_bytes_cycle = st.session_state.get("cycle_video_clean_bytes")
            annotated_bytes_cycle = st.session_state.get("annotated_video_bytes")
            selected_subject_cycle = st.session_state.get("selected_subject_label_cam1")

        # Fallback de compatibilidad: si todavía existe analysis_df, reconstruir una vez.
        if not _valid_dataframe(seg_cycle):
            df_fallback = st.session_state.get("analysis_df")
            if _valid_dataframe(df_fallback):
                seg_cycle = df_fallback.copy()
                snap_fb = _make_cycle_snapshot(seg_cycle, fps_cycle)
                if snap_fb:
                    seg_cycle = snap_fb["seg"]
                    Lsum = snap_fb["left"]
                    Rsum = snap_fb["right"]
                    st.session_state.cycle_seg = seg_cycle
                    st.session_state.cycle_support_left = Lsum
                    st.session_state.cycle_support_right = Rsum
                    st.session_state.cycle_fps = fps_cycle

        if not _valid_dataframe(seg_cycle):
            st.warning(
                "No hay snapshot del segmento 2D. Este registro fue calculado antes de v0.10.1 "
                "o se perdió la sesión de Streamlit. Recalcula una vez el registro; a partir de entonces "
                "la pestaña 9 conservará el segmento aunque se eliminen los vídeos/JSON temporales."
            )
        else:
            try:
                if not isinstance(Lsum, dict) or not isinstance(Rsum, dict):
                    snap_rebuild = _make_cycle_snapshot(seg_cycle, fps_cycle)
                    Lsum = snap_rebuild["left"] if snap_rebuild else {}
                    Rsum = snap_rebuild["right"] if snap_rebuild else {}

                left_cycles = _build_side_cycles(Lsum, fps_cycle, "L")
                right_cycles = _build_side_cycles(Rsum, fps_cycle, "R")

                if selected_subject_cycle:
                    st.success(
                        f"🔒 Sujeto biomecánico bloqueado: **{selected_subject_cycle}**"
                    )

                if not left_cycles and not right_cycles:
                    st.warning("No se han podido construir ciclos IC→TO→IC válidos.")
                else:
                    # v0.10.7 · navegación en SEGUNDOS y mapa temporal de fases.
                    frame_series = pd.to_numeric(seg_cycle["frame"], errors="coerce")
                    fmin = int(np.nanmin(frame_series))
                    fmax = int(np.nanmax(frame_series))
                    duration_s = max(0.0, (fmax-fmin)/fps_cycle)

                    # Estado maestro en segundos para que la barra tenga significado clínico.
                    time_key = "cycle_master_time_s"
                    if time_key not in st.session_state:
                        st.session_state[time_key] = 0.0
                    st.session_state[time_key] = float(
                        np.clip(st.session_state[time_key], 0.0, duration_s)
                    )

                    # Barra MAESTRA bajo el vídeo.
                    current_time_s = float(st.session_state[time_key])
                    current_frame = int(round(fmin + current_time_s*fps_cycle))
                    current_frame = int(np.clip(current_frame, fmin, fmax))

                    left_cycle = _find_cycle_for_frame(left_cycles, current_frame)
                    right_cycle = _find_cycle_for_frame(right_cycles, current_frame)
                    left_phase, left_pct = _phase_context_text(left_cycles, current_frame)
                    right_phase, right_pct = _phase_context_text(right_cycles, current_frame)

                    video_col, phase_col = st.columns([0.72,1.28], gap="medium")

                    with video_col:
                        st.caption("Vídeo + tracking")
                        frame_bytes = None
                        if clean_bytes_cycle:
                            clean_frame = _video_frame_from_bytes(clean_bytes_cycle, current_frame)
                            frame_bytes = _draw_selected_tracking(
                                clean_frame, current_frame, seg_cycle,
                                left_cycle, right_cycle, crop_subject=True
                            )
                        else:
                            video_path_cycle = (
                                st.session_state.get("video2_path")
                                if cycle_view_choice=="Lateral"
                                else st.session_state.get("video1_path")
                            )
                            if video_path_cycle and Path(video_path_cycle).exists():
                                cap_tmp=cv2.VideoCapture(str(video_path_cycle))
                                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES,current_frame)
                                ok_tmp,clean_frame=cap_tmp.read()
                                cap_tmp.release()
                                if ok_tmp:
                                    frame_bytes=_draw_selected_tracking(
                                        clean_frame,current_frame,seg_cycle,
                                        left_cycle,right_cycle,crop_subject=True
                                    )

                        if frame_bytes:
                            st.image(frame_bytes, width=360)
                        else:
                            st.info("Vídeo no disponible; las curvas permanecen accesibles.")

                        st.markdown("**Desplazar por la marcha**")
                        current_time_s = st.slider(
                            "Tiempo del segmento",
                            min_value=0.0,
                            max_value=float(duration_s),
                            value=float(st.session_state[time_key]),
                            step=float(1.0/fps_cycle),
                            format="%.2f s",
                            key=time_key,
                            help="Mueve esta barra: vídeo, mapa de fases y curvas cambian al mismo instante."
                        )
                        current_frame = int(round(fmin + current_time_s*fps_cycle))
                        current_frame = int(np.clip(current_frame, fmin, fmax))

                        # Navegación rápida por eventos anatómicos.
                        events = _all_event_frames(left_cycles, right_cycles)
                        prev_events = [e for e in events if e[0] < current_frame]
                        next_events = [e for e in events if e[0] > current_frame]
                        b1,b2 = st.columns(2)
                        if b1.button("◀ Evento anterior", use_container_width=True):
                            if prev_events:
                                target = prev_events[-1][0]
                                st.session_state[time_key] = (target-fmin)/fps_cycle
                                st.rerun()
                        if b2.button("Evento siguiente ▶", use_container_width=True):
                            if next_events:
                                target = next_events[0][0]
                                st.session_state[time_key] = (target-fmin)/fps_cycle
                                st.rerun()

                        # Contexto textual inmediato.
                        left_phase, left_pct = _phase_context_text(left_cycles, current_frame)
                        right_phase, right_pct = _phase_context_text(right_cycles, current_frame)
                        st.caption(
                            f"{current_time_s:.2f} s · frame {current_frame}  |  "
                            f"Izq: {left_phase}"
                            + (f" ({left_pct:.0f}%)" if np.isfinite(left_pct) else "")
                            + "  ·  "
                            f"Der: {right_phase}"
                            + (f" ({right_pct:.0f}%)" if np.isfinite(right_pct) else "")
                        )

                    # Recalcular tras el widget.
                    left_cycle = _find_cycle_for_frame(left_cycles, current_frame)
                    right_cycle = _find_cycle_for_frame(right_cycles, current_frame)
                    left_phase, left_pct = _phase_context_text(left_cycles, current_frame)
                    right_phase, right_pct = _phase_context_text(right_cycles, current_frame)

                    with phase_col:
                        mm1,mm2,mm3 = st.columns(3)
                        mm1.metric("Izquierda", left_phase)
                        mm2.metric("Derecha", right_phase)
                        mm3.metric("Tiempo", f"{current_time_s:.2f} s")

                        # Mapa temporal GLOBAL: mismo eje temporal que el slider.
                        fig_timeline = _whole_segment_phase_timeline(
                            left_cycles, right_cycles, fps_cycle,
                            fmin, fmax, current_frame
                        )
                        st.pyplot(fig_timeline, use_container_width=True)
                        bio_timeline = io.BytesIO()
                        fig_timeline.savefig(bio_timeline, format="png", dpi=160, bbox_inches="tight")
                        plt.close(fig_timeline)
                        st.session_state["cycle_phase_png"] = bio_timeline.getvalue()

                        kin = _compute_display_kinematics(seg_cycle)
                        numeric_candidates = [
                            c for c in kin.columns
                            if c not in ("frame","time_s")
                            and pd.api.types.is_numeric_dtype(kin[c])
                            and any(k in c for k in (
                                "Oblicuidad","Flexión","Ángulo tobillo","Inclinación",
                                "Desviación frontal","Orientación pie","Retropié"
                            ))
                            and pd.to_numeric(kin[c],errors="coerce").notna().sum() >= 2
                        ]

                        if not numeric_candidates:
                            st.warning("No hay señales cinemáticas suficientes para graficar.")
                        else:
                            dli = next((i for i,c in enumerate(numeric_candidates) if "Izquierda" in c),0)
                            dri = next((i for i,c in enumerate(numeric_candidates) if "Derecha" in c),min(1,len(numeric_candidates)-1))
                            q1,q2 = st.columns(2)
                            var_l = q1.selectbox("Curva izquierda", numeric_candidates, index=dli, key="cycle_var_l")
                            var_r = q2.selectbox("Curva derecha", numeric_candidates, index=dri, key="cycle_var_r")

                            # Vista principal: MISMO eje temporal que el mapa de fases.
                            fig_kin_time = _whole_segment_kinematic_figure(
                                kin, var_l, var_r, fmin, fps_cycle, current_frame
                            )
                            st.pyplot(fig_kin_time, use_container_width=True)
                            bio = io.BytesIO()
                            fig_kin_time.savefig(bio,format="png",dpi=170,bbox_inches="tight")
                            plt.close(fig_kin_time)
                            st.session_state["cycle_kinematic_png"] = bio.getvalue()

                            # Vista 0–100% solo como análisis secundario del ciclo actual.
                            with st.expander("Curva normalizada 0–100% del ciclo actual", expanded=False):
                                df_l = _normalized_cycle_series(kin,left_cycle,[var_l]) if left_cycle else pd.DataFrame()
                                df_r = _normalized_cycle_series(kin,right_cycle,[var_r]) if right_cycle else pd.DataFrame()
                                fig_kin_norm = _kinematic_cycle_figure(
                                    df_l,df_r,var_l,var_r,
                                    left_pct=left_pct,right_pct=right_pct
                                )
                                st.pyplot(fig_kin_norm,use_container_width=True)
                                plt.close(fig_kin_norm)


                    st.markdown("### Reproducción continua")
                    st.caption(
                        "Reproduce el segmento completo con las fases izquierda/derecha "
                        "sobreimpresas. Selecciona una velocidad para cámara lenta o rápida."
                    )

                    speed_col, action_col = st.columns([0.45,0.55])
                    with speed_col:
                        playback_speed = st.selectbox(
                            "Velocidad",
                            [0.25,0.5,1.0,1.5,2.0],
                            index=2,
                            format_func=lambda x: f"{x:g}×",
                            key="cycle_playback_speed",
                        )
                    with action_col:
                        st.write("")
                        prepare_playback = st.button(
                            "▶ Preparar reproducción",
                            use_container_width=True,
                            key="prepare_cycle_playback",
                        )

                    # Construir la versión base solo cuando se solicita.
                    if prepare_playback:
                        source_playback = annotated_bytes_cycle or clean_bytes_cycle
                        if source_playback:
                            with st.spinner("Preparando vídeo de revisión…"):
                                base_phase_video = _build_phase_review_video_bytes(
                                    source_playback, left_cycles, right_cycles, fps_cycle
                                )
                                st.session_state["cycle_phase_review_base"] = base_phase_video
                                st.session_state["cycle_phase_review_speed"] = None
                        else:
                            st.warning("No queda vídeo disponible en memoria para preparar la reproducción.")

                    base_phase_video = st.session_state.get("cycle_phase_review_base")
                    if base_phase_video:
                        cached_speed = st.session_state.get("cycle_phase_review_speed")
                        if cached_speed != playback_speed or not st.session_state.get("cycle_phase_review_bytes"):
                            with st.spinner(f"Preparando reproducción a {playback_speed:g}×…"):
                                speed_bytes = _change_video_speed_bytes(
                                    base_phase_video, playback_speed
                                )
                                st.session_state["cycle_phase_review_bytes"] = speed_bytes or base_phase_video
                                st.session_state["cycle_phase_review_speed"] = playback_speed

                        st.video(st.session_state["cycle_phase_review_bytes"])
                        st.caption(
                            "Este reproductor es para inspección continua. "
                            "Para comprobar un instante exacto usa la barra temporal superior."
                        )

                    with st.expander("⚙️ Corrección manual del ciclo actual",expanded=False):
                        st.caption("Solo si IC/TO no coincide visualmente. No vuelve a ejecutar Pose2Sim.")
                        e1,e2=st.columns(2)
                        with e1:
                            st.markdown("**Izquierda**")
                            left_cycle=_manual_event_editor(left_cycle,f"manual_L_auto_{left_cycle['cycle_index'] if left_cycle else 0}") if left_cycle else None
                        with e2:
                            st.markdown("**Derecha**")
                            right_cycle=_manual_event_editor(right_cycle,f"manual_R_auto_{right_cycle['cycle_index'] if right_cycle else 0}") if right_cycle else None

                        with st.expander("Resumen numérico del ciclo actual", expanded=False):

                            # Table with phase summary
                            rows=[]
                            for cyc,label in [(left_cycle,"Izquierda"),(right_cycle,"Derecha")]:
                                if not cyc:
                                    continue
                                cyc_frames=max(1,cyc["next_ic_frame"]-cyc["ic_frame"])
                                stance_pct=float(cyc.get("stance_pct",np.nan))
                                swing_pct=100.0-stance_pct if np.isfinite(stance_pct) else np.nan
                                rows.append({
                                    "Extremidad":label,
                                    "IC frame":cyc["ic_frame"],
                                    "TO frame":cyc.get("to_frame"),
                                    "IC siguiente":cyc["next_ic_frame"],
                                    "Ciclo (s)":cyc_frames/fps_cycle,
                                    "Apoyo (%)":stance_pct,
                                    "Swing (%)":swing_pct,
                                    "Apoyo (s)":((cyc.get("to_frame")-cyc["ic_frame"])/fps_cycle) if cyc.get("to_frame") else np.nan,
                                    "Swing (s)":((cyc["next_ic_frame"]-cyc.get("to_frame"))/fps_cycle) if cyc.get("to_frame") else np.nan,
                                })
                            if rows:
                                phase_df=pd.DataFrame(rows)
                                st.dataframe(phase_df,use_container_width=True,hide_index=True)
                                st.session_state["cycle_phase_table"]=phase_df

                        st.caption(
                            "IC/TO = eventos cinemáticos 2D estimados; no equivalen a fuerza de impacto."
                        )

            except Exception as e:
                st.error(f"No se pudo construir el analizador de ciclo: {e}")
                with st.expander("Detalles técnicos"):
                    st.code(traceback.format_exc())


st.divider()
st.caption("PhysioSentinel Gait v0.10.9 · IC/TO con QC bilateral · mapa/cinemática alineados · reproducción 0.25×–2× · exportación")
