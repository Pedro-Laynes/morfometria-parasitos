import io
import zipfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import sato
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Morfometría Taxonómica de Nematodos",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 10px 14px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CATÁLOGOS TAXONÓMICOS, VIDRIERÍA Y PERFILES ANATÓMICOS
# ==============================================================================
MODOS_CALIBRACION = [
    "Microscopio / ToupView (Barra de escala de 2 puntos en µm)",
    "Portaobjetos estándar (76.2 × 25.4 mm - 4 esquinas)",
    "Laminilla cubreobjetos (20.0 × 20.0 mm - 4 esquinas)",
    "Laminilla cubreobjetos (22.0 × 22.0 mm - 4 esquinas)",
    "Laminilla cubreobjetos (24.0 × 24.0 mm - 4 esquinas)",
    "Laminilla cubreobjetos (18.0 × 18.0 mm - 4 esquinas)",
    "Laminilla rectangular (24.0 × 50.0 mm - 4 esquinas)",
    "Medida personalizada (4 esquinas)...",
]

DIMS_VIDRIERIA = {
    "Portaobjetos estándar (76.2 × 25.4 mm - 4 esquinas)": (76.2, 25.4, 20.0),
    "Laminilla cubreobjetos (20.0 × 20.0 mm - 4 esquinas)": (20.0, 20.0, 40.0),
    "Laminilla cubreobjetos (22.0 × 22.0 mm - 4 esquinas)": (22.0, 22.0, 40.0),
    "Laminilla cubreobjetos (24.0 × 24.0 mm - 4 esquinas)": (24.0, 24.0, 40.0),
    "Laminilla cubreobjetos (18.0 × 18.0 mm - 4 esquinas)": (18.0, 18.0, 45.0),
    "Laminilla rectangular (24.0 × 50.0 mm - 4 esquinas)": (50.0, 24.0, 25.0),
}

# (Código, Descripción, Color Hex, Tipo: 'cuerpo_auto' | 'eje' | 'ancho_directo')
CATALOGO_ESTRUCTURAS = {
    "LT":   ("Longitud Total [LT] (y ACM automático en foto macro)", "#ff2a2a", "cuerpo_auto"),
    "ACM":  ("Ancho Corporal Máximo [ACM] (2 clics pared a pared)", "#ffdd00", "ancho_directo"),
    "AN":   ("Longitud Anillo Nervioso a Extremo Anterior [AN]", "#00e5ff", "eje"),
    "LE":   ("Longitud del Esófago [LE] (hasta ventrículo)", "#ff9100", "eje"),
    "AE":   ("Ancho del Esófago [AE] (2 clics pared a pared)", "#ffb74d", "ancho_directo"),
    "LV":   ("Longitud del Ventrículo [LV]", "#d500f9", "eje"),
    "AV":   ("Ancho del Ventrículo [AV] (2 clics pared a pared)", "#ea80fc", "ancho_directo"),
    "LAV":  ("Longitud del Apéndice Ventricular [LAV]", "#00e676", "eje"),
    "AAV":  ("Ancho del Apéndice Ventricular [AAV] (2 clics)", "#69f0ae", "ancho_directo"),
    "LCI":  ("Longitud del Ciego Intestinal [LCI]", "#ffea00", "eje"),
    "ACI":  ("Ancho del Ciego Intestinal [ACI] (2 clics)", "#fff59d", "ancho_directo"),
    "LC":   ("Longitud de la Cola [LC] (ano/cloaca a extremo posterior)", "#2979ff", "eje"),
    "ACA":  ("Ancho Corporal a nivel del Ano/Cloaca [ACA] (2 clics)", "#82b1ff", "ancho_directo"),
    "DV":   ("Distancia Cabeza a Vulva [DV] (Adulto - Hembra)", "#ff4081", "eje"),
    "LESP": ("Longitud de Espículas [LESP] (Adulto - Macho)", "#76ff03", "eje"),
}

# Perfiles anatómicos típicos según género (estructuras presentes por defecto)
PERFILES_GENERO = {
    "Personalizado (Todas habilitadas)": list(CATALOGO_ESTRUCTURAS.keys()),
    "Anisakis sp. (Sin ciego ni apéndice)": ["LT", "ACM", "AN", "LE", "AE", "LV", "AV", "LC", "ACA", "DV", "LESP"],
    "Contracaecum sp. (Con ciego y apéndice)": ["LT", "ACM", "AN", "LE", "AE", "LV", "AV", "LAV", "AAV", "LCI", "ACI", "LC", "ACA", "DV", "LESP"],
    "Pseudoterranova sp. (Con ciego, sin apéndice)": ["LT", "ACM", "AN", "LE", "AE", "LV", "AV", "LCI", "ACI", "LC", "ACA", "DV", "LESP"],
    "Hysterothylacium sp. (Con ciego y apéndice)": ["LT", "ACM", "AN", "LE", "AE", "LV", "AV", "LAV", "AAV", "LCI", "ACI", "LC", "ACA", "DV", "LESP"],
}

ANCHO_VISOR_PX = 950


# ==============================================================================
# FUNCIONES DE CALIBRACIÓN ÓPTICA Y HOMOGRAFÍA
# ==============================================================================
def corregir_distorsion_radial(img_bgr, k1):
    if abs(k1) < 1e-4:
        return img_bgr
    h, w = img_bgr.shape[:2]
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float32)
    D = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32)
    return cv2.undistort(img_bgr, K, D)


def ordenar_esquinas(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def dibujar_regla_calibrada(img, w_mm, h_mm, px_per_mm):
    h, w = img.shape[:2]
    color_regla = (0, 255, 255)
    color_texto = (0, 255, 0)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color_regla, 2)
    paso_etiqueta = 5 if max(w_mm, h_mm) <= 30.0 else 10

    for mm in range(0, int(np.floor(w_mm)) + 1):
        x = int(round(mm * px_per_mm))
        largo = 20 if mm % paso_etiqueta == 0 else (13 if mm % 5 == 0 else 7)
        grosor = 2 if mm % paso_etiqueta == 0 else 1
        if mm % paso_etiqueta == 0 and mm > 0:
            cv2.putText(img, f"{mm}", (x - 10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_texto, 1, cv2.LINE_AA)
        cv2.line(img, (x, 0), (x, largo), color_regla, grosor)
        cv2.line(img, (x, h - 1), (x, h - 1 - largo), color_regla, grosor)

    for mm in range(0, int(np.floor(h_mm)) + 1):
        y = int(round(mm * px_per_mm))
        largo = 20 if mm % paso_etiqueta == 0 else (13 if mm % 5 == 0 else 7)
        grosor = 2 if mm % paso_etiqueta == 0 else 1
        if mm % paso_etiqueta == 0 and mm > 0:
            cv2.putText(img, f"{mm}", (24, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_texto, 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (largo, y), color_regla, grosor)
        cv2.line(img, (w - 1, y), (w - 1 - largo, y), color_regla, grosor)
    return img


def rectificar_lamina(img_bgr, puntos_esquinas, dim1_mm, dim2_mm, px_per_mm):
    esquinas_src = ordenar_esquinas(puntos_esquinas)
    ancho_px = np.linalg.norm(esquinas_src[1] - esquinas_src[0])
    alto_px = np.linalg.norm(esquinas_src[3] - esquinas_src[0])

    lado_mayor, lado_menor = max(dim1_mm, dim2_mm), min(dim1_mm, dim2_mm)
    if abs(lado_mayor - lado_menor) < 1e-3:
        w_mm, h_mm = lado_mayor, lado_menor
    else:
        w_mm, h_mm = (lado_mayor, lado_menor) if ancho_px >= alto_px else (lado_menor, lado_mayor)

    dst_w = int(round(w_mm * px_per_mm))
    dst_h = int(round(h_mm * px_per_mm))

    esquinas_dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(esquinas_src, esquinas_dst)
    img_rect = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC)
    img_regla = dibujar_regla_calibrada(img_rect.copy(), w_mm, h_mm, px_per_mm)
    return img_rect, img_regla


# ==============================================================================
# SEGMENTACIÓN Y LIMPIEZA INTERACTIVA DE ARTEFACTOS
# ==============================================================================
def segmentar_cuerpo_macro(img_rect_bgr, gamma_fondo, ganancia_gusano, umbral, px_per_mm,
                           puntos_guia=None, radio_corredor_mm=0.6, usar_sato=False,
                           puntos_borrar=None):
    canal_L = cv2.cvtColor(img_rect_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
    L_u8 = (canal_L * 255).astype(np.uint8)

    k_size = max(15, int(round(1.5 * px_per_mm)) | 1)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(L_u8, cv2.MORPH_TOPHAT, kernel_tophat).astype(np.float32) / 255.0
    base_L = np.clip((canal_L ** gamma_fondo) * ganancia_gusano + 0.40 * tophat, 0.0, 1.0)

    if usar_sato:
        sigmas = np.linspace(1.5, max(3.0, 0.35 * px_per_mm), 4)
        tubular = sato(canal_L, sigmas=sigmas, black_ridges=False)
        if tubular.max() > 0:
            tubular = tubular / tubular.max()
        base_L = np.clip(0.55 * base_L + 0.65 * tubular, 0.0, 1.0)

    img_realzada = (base_L * 255).astype(np.uint8)
    mascara = (img_realzada >= umbral).astype(bool)

    margen_px = max(8, int(round(0.6 * px_per_mm)))
    mascara[:margen_px, :] = False
    mascara[-margen_px:, :] = False
    mascara[:, :margen_px] = False
    mascara[:, -margen_px:] = False

    if puntos_guia is not None and len(puntos_guia) >= 2:
        roi = np.zeros(mascara.shape, dtype=np.uint8)
        grosor_px = max(8, int(round(2.0 * radio_corredor_mm * px_per_mm)))
        cv2.polylines(roi, [np.array(puntos_guia, dtype=np.int32)], isClosed=False, color=255, thickness=grosor_px)
        mascara = mascara & (roi > 0)

    mascara = remove_small_objects(mascara, min_size=max(15, int(0.08 * px_per_mm ** 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_u8 = cv2.morphologyEx(mascara.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    mascara_u8 = cv2.morphologyEx(mascara_u8, cv2.MORPH_OPEN, kernel)

    # Eliminar formas / artefactos donde el usuario hizo clic en modo borrador
    if puntos_borrar:
        num_labels, labels = cv2.connectedComponents(mascara_u8)
        h_m, w_m = mascara_u8.shape
        radio_busq = max(4, int(round(0.25 * px_per_mm)))
        for bx, by in puntos_borrar:
            ix = int(np.clip(round(bx), 0, w_m - 1))
            iy = int(np.clip(round(by), 0, h_m - 1))
            # Buscar etiqueta en el punto exacto o vecindad inmediata
            y1, y2 = max(0, iy - radio_busq), min(h_m, iy + radio_busq + 1)
            x1, x2 = max(0, ix - radio_busq), min(w_m, ix + radio_busq + 1)
            parche = labels[y1:y2, x1:x2]
            etiquetas_tocadas = np.unique(parche[parche > 0])
            for et in etiquetas_tocadas:
                mascara_u8[labels == et] = 0
                # También borrar un radio local por si estaba pegado al gusano
                cv2.circle(mascara_u8, (ix, iy), max(6, int(round(0.35 * px_per_mm))), 0, -1)

    return img_realzada, mascara_u8


def mejorar_contraste_microscopio(img_bgr, gamma_val, gain_val):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L).astype(np.float32) / 255.0
    L_adj = np.clip((L_clahe ** (gamma_val / 2.2)) * gain_val, 0.0, 1.0)
    lab_adj = cv2.merge([(L_adj * 255).astype(np.uint8), a, b])
    return cv2.cvtColor(lab_adj, cv2.COLOR_LAB2RGB)


def suavizar_curva_subpixel(ruta_xy, n_puntos=300):
    pts = np.array(ruta_xy, dtype=np.float64)
    if len(pts) < 4:
        return pts
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = np.vstack([pts[0], pts[1:][diffs > 0.2]])
    if len(pts) < 4:
        return pts
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 0.30, k=3)
        u_fine = np.linspace(0, 1, n_puntos)
        xs, ys = splev(u_fine, tck)
        curva = np.column_stack((xs, ys))
        curva[0] = pts[0]
        curva[-1] = pts[-1]
        return curva
    except Exception:
        return pts


def medir_estructura(img_bgr, mascara_macro, puntos_guia, px_per_mm, tipo_estructura,
                     es_microscopio=False, suavizar_subpixel=True):
    pts = np.array(puntos_guia, dtype=np.float64)
    p_inicio, p_fin = pts[0], pts[-1]

    if tipo_estructura == "ancho_directo":
        pasos = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.array([0.0])
        dist_mm = float(np.sum(pasos) / px_per_mm)
        return {
            "valor_mm": dist_mm,
            "ancho_auto_mm": 0.0,
            "top3_anchos_mm": [],
            "ruta_xy": pts,
            "segmento_ancho": None,
            "puntos_originales": list(puntos_guia),
            "tipo": tipo_estructura,
        }

    if es_microscopio or tipo_estructura == "eje" or mascara_macro is None:
        segmentos = []
        for i in range(len(pts) - 1):
            t = np.linspace(0, 1, 80, endpoint=(i == len(pts) - 2))
            segmentos.append(pts[i] + np.outer(t, pts[i + 1] - pts[i]))
        ruta_xy = np.vstack(segmentos) if len(segmentos) > 0 else pts
        if suavizar_subpixel and len(pts) >= 4:
            ruta_xy = suavizar_curva_subpixel(ruta_xy)
        pasos = np.hypot(*np.diff(ruta_xy, axis=0).T) if len(ruta_xy) > 1 else np.array([0.0])
        long_mm = float(np.sum(pasos) / px_per_mm)
        return {
            "valor_mm": long_mm,
            "ancho_auto_mm": 0.0,
            "top3_anchos_mm": [],
            "ruta_xy": ruta_xy,
            "segmento_ancho": None,
            "puntos_originales": list(puntos_guia),
            "tipo": tipo_estructura,
        }

    mask_bin = (mascara_macro > 0).astype(np.uint8)
    h_img, w_img = mask_bin.shape
    esqueleto = skeletonize(mask_bin > 0)
    skel_ys, skel_xs = np.where(esqueleto)
    skel_coords = np.column_stack((skel_xs, skel_ys))

    if len(skel_coords) >= 8:
        costo = np.full((h_img, w_img), 500.0, dtype=np.float64)
        costo[mask_bin > 0] = 20.0
        costo[esqueleto] = 1.0

        nodos_rc = []
        for pt in puntos_guia:
            pt_arr = np.array(pt, dtype=np.float32)
            dists = np.linalg.norm(skel_coords - pt_arr, axis=1)
            idx_c = int(np.argmin(dists))
            if dists[idx_c] < (0.75 * px_per_mm):
                nodos_rc.append((int(skel_ys[idx_c]), int(skel_xs[idx_c])))
            else:
                nodos_rc.append((int(np.clip(round(pt[1]), 0, h_img - 1)),
                                 int(np.clip(round(pt[0]), 0, w_img - 1))))

        tramos_rc = []
        for i in range(len(nodos_rc) - 1):
            seg_rc, _ = route_through_array(costo, nodos_rc[i], nodos_rc[i + 1], fully_connected=True)
            seg_arr = np.array(seg_rc)
            if i > 0 and len(seg_arr) > 1:
                seg_arr = seg_arr[1:]
            tramos_rc.append(seg_arr)
        ruta_xy = np.vstack(tramos_rc)[:, ::-1].astype(np.float64)
    else:
        segmentos = []
        for i in range(len(pts) - 1):
            t = np.linspace(0, 1, 80, endpoint=(i == len(pts) - 2))
            segmentos.append(pts[i] + np.outer(t, pts[i + 1] - pts[i]))
        ruta_xy = np.vstack(segmentos)

    if len(ruta_xy) > 4:
        if np.linalg.norm(p_inicio - ruta_xy[0]) > 1.5:
            ruta_xy = np.vstack([p_inicio, ruta_xy[2:]])
        if np.linalg.norm(p_fin - ruta_xy[-1]) > 1.5:
            ruta_xy = np.vstack([ruta_xy[:-2], p_fin])

    if suavizar_subpixel:
        ruta_xy = suavizar_curva_subpixel(ruta_xy)

    pasos = np.hypot(*np.diff(ruta_xy, axis=0).T) if len(ruta_xy) > 1 else np.array([0.0])
    long_mm = float(np.sum(pasos) / px_per_mm)

    ancho_max_mm, top3_anchos_mm, segmento_ancho = 0.0, [], None
    if np.sum(mask_bin) > 15:
        mapa_radios = distance_transform_edt(mask_bin)
        ruta_rc_eval = np.column_stack((
            np.clip(np.round(ruta_xy[:, 1]).astype(int), 0, h_img - 1),
            np.clip(np.round(ruta_xy[:, 0]).astype(int), 0, w_img - 1)
        ))
        margen_ext = max(1, int(len(ruta_rc_eval) * 0.05))
        central_rc = ruta_rc_eval[margen_ext:-margen_ext] if len(ruta_rc_eval) > 20 else ruta_rc_eval
        radios = mapa_radios[central_rc[:, 0], central_rc[:, 1]]

        if len(radios) > 0:
            dist_min_picos = max(1, len(radios) // 8)
            picos, _ = find_peaks(radios, distance=dist_min_picos)
            if len(picos) == 0:
                picos = np.array([int(np.argmax(radios))])
            picos_ord = picos[np.argsort(radios[picos])[::-1]][:3]
            top3_anchos_mm = [round(float(2.0 * radios[p] / px_per_mm), 3) for p in picos_ord]

            idx_max = int(picos_ord[0])
            radio_max_px = float(radios[idx_max])
            ancho_max_mm = float((2.0 * radio_max_px) / px_per_mm)

            centro_xy = central_rc[idx_max][::-1]
            idx_glob = margen_ext + idx_max
            tangente = ruta_xy[min(len(ruta_xy) - 1, idx_glob + 3)] - ruta_xy[max(0, idx_glob - 3)]
            norma_t = np.linalg.norm(tangente)
            normal = np.array([-tangente[1], tangente[0]]) / norma_t if norma_t > 0 else np.array([0.0, 1.0])
            segmento_ancho = (centro_xy + normal * radio_max_px, centro_xy - normal * radio_max_px)

    return {
        "valor_mm": long_mm,
        "ancho_auto_mm": ancho_max_mm,
        "top3_anchos_mm": top3_anchos_mm,
        "ruta_xy": ruta_xy,
        "segmento_ancho": segmento_ancho,
        "puntos_originales": list(puntos_guia),
        "tipo": tipo_estructura,
    }


def calcular_indices_taxonomicos(medidas):
    def ratio(num_key, den_key, factor=1.0):
        num = medidas.get(num_key)
        den = medidas.get(den_key)
        if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den > 0:
            return round((num / den) * factor, 2)
        return None

    modelo_personalizado = {
        "LT/ACM": ratio("LT", "ACM"),
        "LT/LE": ratio("LT", "LE"),
        "LT/LCI": ratio("LT", "LCI"),
        "LT/LAV": ratio("LT", "LAV"),
        "LV/LCI": ratio("LV", "LCI"),
        "LAV/LCI": ratio("LAV", "LCI"),
        "LV/LE (LV/E)": ratio("LV", "LE"),
        "AN/LE (%)": ratio("AN", "LE", 100.0),
        "AE/LE": ratio("AE", "LE"),
        "AV/LV": ratio("AV", "LV"),
        "AAV/LAV": ratio("AAV", "LAV"),
        "ACI/LCI": ratio("ACI", "LCI"),
    }

    le = medidas.get("LE") if isinstance(medidas.get("LE"), (int, float)) else None
    lv = medidas.get("LV") if isinstance(medidas.get("LV"), (int, float)) else None
    lt = medidas.get("LT") if isinstance(medidas.get("LT"), (int, float)) else None
    le_gland = (le + lv) if (le is not None and lv is not None) else None
    b_prima = round(lt / le_gland, 2) if (lt and le_gland and le_gland > 0) else None

    indice_deman = {
        "DeMan_a (LT/ACM)": ratio("LT", "ACM"),
        "DeMan_b (LT/LE)": ratio("LT", "LE"),
        "DeMan_b' (LT/[LE+LV])": b_prima,
        "DeMan_c (LT/LC)": ratio("LT", "LC"),
        "DeMan_c' (LC/ACA)": ratio("LC", "ACA"),
        "DeMan_V% (DV/LT*100)": ratio("DV", "LT", 100.0),
    }
    return modelo_personalizado, indice_deman


def generar_lamina_validacion(img_rgb, trazos_guardados, medidas_dict, indices_activos,
                              titulo_ficha, meta_dict, px_per_mm):
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.imshow(img_rgb)

    for cod, trazo in trazos_guardados.items():
        _, color_hex, tipo = CATALOGO_ESTRUCTURAS[cod]
        ruta = trazo["ruta_xy"]
        estilo_linea = "--" if tipo == "ancho_directo" else "-"
        val_um = trazo["valor_mm"] * 1000.0
        ax.plot(ruta[:, 0], ruta[:, 1], color=color_hex, linestyle=estilo_linea, linewidth=2.2,
                label=f"{cod}: {trazo['valor_mm']:.3f} mm ({val_um:.0f} µm)")
        ax.scatter([ruta[0, 0], ruta[-1, 0]], [ruta[0, 1], ruta[-1, 1]],
                   color=color_hex, edgecolors="black", s=40, zorder=6)

        if trazo["segmento_ancho"] is not None and trazo["ancho_auto_mm"] > 0:
            p1, p2 = trazo["segmento_ancho"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ffdd00", linestyle="--", linewidth=2.2,
                    label=f"ACM: {trazo['ancho_auto_mm']:.3f} mm")

    lineas_texto = [
        f"ESPÉCIMEN: {meta_dict['ID_Especimen']} | FOTO: {titulo_ficha}",
        f"TAXÓN: {meta_dict['Genero']} ({meta_dict['Morfotipo']}) | {meta_dict['Estadio']} | Hosp: {meta_dict['Hospedador']}",
        f"Escala: {px_per_mm:.2f} px/mm ({px_per_mm/1000.0:.3f} px/µm)",
    ]
    resumen_idx = [f"{k}={v}" for k, v in indices_activos.items() if v is not None][:6]
    if resumen_idx:
        lineas_texto.append("ÍNDICES: " + " | ".join(resumen_idx))

    ax.text(
        0.015, 0.96, "\n".join(lineas_texto), transform=ax.transAxes, verticalalignment="top",
        color="white", fontsize=9, fontweight="bold", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#111111", edgecolor="#00ff88", alpha=0.88)
    )

    ax.set_title(f"Ficha Morfométrica — {titulo_ficha}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.88, fontsize=8.5)
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# ESTADO DE SESIÓN MULTI-IMAGEN POR ESPÉCIMEN
# ==============================================================================
defaults = {
    "foto_activa_idx": 0,
    "estado_por_foto": {},       # {nombre_foto: {'cal_pts': [], 'modo_cal': str, 'escala_um': float|None, 'trazos': {}, 'borrar_pts': [], 'top3': []}}
    "ultimo_clic_cal": None,
    "extremos_pts": [],
    "ultimo_clic_ext": None,
    "ultimo_clic_borrador": None,
    "estructuras_ausentes": set(),
    "base_especimenes": {},      # {ID_Especimen: dict_consolidado}
    "imagenes_validadas": {},    # {ID_Especimen__Foto: bytes_png}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def obtener_estado_foto(nombre_foto):
    if nombre_foto not in st.session_state.estado_por_foto:
        st.session_state.estado_por_foto[nombre_foto] = {
            "cal_pts": [],
            "modo_cal": MODOS_CALIBRACION[0],
            "escala_um": None,   # None hasta que el usuario ingrese el valor real de la barra
            "trazos": {},
            "borrar_pts": [],
            "top3": [],
        }
    return st.session_state.estado_por_foto[nombre_foto]


# ==============================================================================
# BARRA LATERAL: CARGA, TAXONOMÍA, CARACTERES MORFOLÓGICOS Y EXPORTACIÓN
# ==============================================================================
st.title("🔬 Estación Morfométrica y Taxonómica de Nematodos")

with st.sidebar:
    st.header("1. Fotos del Espécimen / Lote")
    archivos_subidos = st.file_uploader(
        "Sube todas las fotos del individuo (macro + 1–4 fotos ToupView)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("2. Identificación y Perfil Taxonómico")
    id_especimen = st.text_input("ID del Espécimen (Ej. LH_134_M)", value="LH_134_M")
    perfil_gen = st.selectbox("Perfil anatómico de referencia:", list(PERFILES_GENERO.keys()), index=0)
    meta_genero = st.text_input("Género / Especie asignada", value="Anisakis sp." if "Anisakis" in perfil_gen else "Contracaecum sp.")
    meta_morfotipo = st.selectbox("Morfotipo", ["Tipo I", "Tipo II", "Tipo III", "Morfotipo A", "Morfotipo B", "No aplica"])
    meta_hospedador = st.text_input("Hospedador", value="No especificado")
    meta_estadio = st.selectbox("Estadio / Sexo", ["Larva (L3/L4)", "Adulto - Hembra", "Adulto - Macho"])

    with st.expander("🧬 Caracteres Morfológicos Cualitativos", expanded=True):
        c_deiridios = st.selectbox("Deiridios", ["No evaluado", "Presentes (prominentes)", "Presentes (inconspicuos)", "Ausentes"])
        c_labios = st.selectbox("Labios cefálicos", ["No evaluado", "Subdesarrollados (3 labios)", "Bien desarrollados (bilobulados)", "Con interlabios"])
        c_cuticula = st.selectbox("Estriación cuticular", ["No evaluado", "Finamente estriada", "Fuertemente estriada", "Anillada / con espinas"])
        c_mucron = st.selectbox("Presencia de mucrón caudal", ["No evaluado", "Presente", "Ausente"])
        c_diente = st.selectbox("Diente larval (bore tooth)", ["No evaluado", "Presente", "Ausente"])
        c_cola = st.selectbox("Forma de la cola", ["No evaluado", "Cónica en punta", "Roma / redondeada", "Ovalada", "Digitiforme / mucronada"])
        c_obs = st.text_input("Observaciones morfológicas", value="")

    st.divider()
    st.header("3. Contraste y Visión Avanzada")
    gamma_val = st.slider("Contraste / Oscurecer Fondo (Gamma)", 1.0, 6.0, 2.2, 0.1)
    gain_val = st.slider("Brillo / Iluminar Tejido (Gain)", 0.8, 3.0, 1.15, 0.05)
    umbral_val = st.slider("Umbral Segmentación Macro (0–255)", 20, 245, 130, 1)
    radio_corredor = st.slider("Radio búsqueda macro (mm)", 0.15, 2.5, 0.6, 0.05)

    with st.expander("🛠️ Visión Avanzada (Sato, Spline y Lente)", expanded=False):
        usar_sato = st.checkbox("Filtro Tubular de Sato (macro translúcido)", value=False)
        suavizar_eje = st.checkbox("Refinamiento Subpíxel de Curva (Spline)", value=True)
        k1_lente = st.slider("Corrección Distorsión Radial Lente (k1)", -0.30, 0.30, 0.0, 0.02)

    st.divider()
    st.header("4. Exportación Consolidada")
    if st.session_state.base_especimenes:
        df_export = pd.DataFrame(list(st.session_state.base_especimenes.values()))
        st.download_button(
            label=f"📊 Descargar CSV Consolidado ({len(df_export)} especímenes)",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="morfometria_taxonomica_consolidada.csv",
            mime="text/csv",
            use_container_width=True
        )
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for id_m, img_bytes in st.session_state.imagenes_validadas.items():
                limpio = id_m.replace(" ", "_").replace("#", "P").replace("/", "_")
                zf.writestr(f"{limpio}_validado.png", img_bytes)
        zip_buf.seek(0)
        st.download_button(
            label=f"🖼️ Descargar Fichas ZIP ({len(st.session_state.imagenes_validadas)})",
            data=zip_buf.getvalue(),
            file_name="fichas_validadas_tesis.zip",
            mime="application/zip",
            use_container_width=True
        )

if not archivos_subidos:
    st.info("👈 Sube tus fotografías (puedes subir simultáneamente la foto macro y hasta 4 fotos de microscopio ToupView del mismo parásito).")
    st.stop()

if st.session_state.foto_activa_idx >= len(archivos_subidos):
    st.session_state.foto_activa_idx = 0

# ==============================================================================
# NAVEGACIÓN SIMULTÁNEA ENTRE LAS FOTOS SUBIDAS DEL ESPÉCIMEN
# ==============================================================================
st.subheader(f"📂 Imágenes cargadas para el Espécimen [{id_especimen}] ({len(archivos_subidos)} archivos)")
cols_tabs = st.columns(min(6, len(archivos_subidos)))
for idx_f, f_obj in enumerate(archivos_subidos):
    est_f = obtener_estado_foto(f_obj.name)
    n_med = len(est_f["trazos"])
    cal_ok = "✅" if (len(est_f["cal_pts"]) in (2, 4)) else "📏"
    etiqueta_btn = f"{cal_ok} Foto {idx_f+1}: {f_obj.name[:16]} ({n_med} med.)"
    tipo_btn = "primary" if idx_f == st.session_state.foto_activa_idx else "secondary"
    if cols_tabs[idx_f % len(cols_tabs)].button(etiqueta_btn, key=f"tab_f_{idx_f}", type=tipo_btn, use_container_width=True):
        st.session_state.foto_activa_idx = idx_f
        st.session_state.extremos_pts = []
        st.session_state.ultimo_clic_ext = None
        st.rerun()

archivo_activo = archivos_subidos[st.session_state.foto_activa_idx]
est_foto = obtener_estado_foto(archivo_activo.name)

bytes_img = np.asarray(bytearray(archivo_activo.read()), dtype=np.uint8)
archivo_activo.seek(0)
img_bgr_raw = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
img_bgr = corregir_distorsion_radial(img_bgr_raw, k1_lente)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h_orig, w_orig = img_rgb.shape[:2]

# ==============================================================================
# PASO 1: CALIBRACIÓN INDEPENDIENTE POR IMAGEN
# ==============================================================================
st.divider()
c_modo1, c_modo2 = st.columns([2.5, 1.5])
with c_modo1:
    modo_sel = st.selectbox(
        f"📐 Referencia de calibración para [{archivo_activo.name}]:",
        options=MODOS_CALIBRACION,
        index=MODOS_CALIBRACION.index(est_foto["modo_cal"]),
        key=f"sel_modo_{archivo_activo.name}"
    )
    if modo_sel != est_foto["modo_cal"]:
        est_foto["modo_cal"] = modo_sel
        est_foto["cal_pts"] = []
        est_foto["escala_um"] = None
        st.rerun()

es_microscopio = "Microscopio" in est_foto["modo_cal"]
pts_req = 2 if es_microscopio else 4

with c_modo2:
    if not es_microscopio:
        if est_foto["modo_cal"] in DIMS_VIDRIERIA:
            dim1_mm, dim2_mm, px_per_mm_vid = DIMS_VIDRIERIA[est_foto["modo_cal"]]
        else:
            d1, d2 = st.columns(2)
            dim1_mm = d1.number_input("Ancho (mm)", 1.0, 200.0, 22.0, 0.5, key=f"d1_{archivo_activo.name}")
            dim2_mm = d2.number_input("Alto (mm)", 1.0, 200.0, 22.0, 0.5, key=f"d2_{archivo_activo.name}")
            px_per_mm_vid = 40.0 if max(dim1_mm, dim2_mm) <= 30.0 else 20.0

if es_microscopio:
    st.subheader("Paso 1: Marca los 2 extremos de la barra de escala e indica su valor real (µm)")
else:
    st.subheader(f"Paso 1: Haz clic en las 4 esquinas de la referencia ({dim1_mm} × {dim2_mm} mm)")

c1, c2 = st.columns([3, 1.1])
with c2:
    if st.button("🔄 Reiniciar Calibración de esta Foto", use_container_width=True):
        est_foto["cal_pts"] = []
        est_foto["escala_um"] = None
        st.session_state.ultimo_clic_cal = None
        st.rerun()
    st.metric("Puntos de calibración", f"{len(est_foto['cal_pts'])} / {pts_req}")

    # En modo microscopio, pedir el valor de la escala una vez observada la foto / marcados los 2 puntos
    if es_microscopio:
        st.markdown("**Valor leído en la barra (µm):**")
        b_s1, b_s2, b_s3, b_s4 = st.columns(4)
        if b_s1.button("50", use_container_width=True):
            est_foto["escala_um"] = 50.0
            st.rerun()
        if b_s2.button("100", use_container_width=True):
            est_foto["escala_um"] = 100.0
            st.rerun()
        if b_s3.button("200", use_container_width=True):
            est_foto["escala_um"] = 200.0
            st.rerun()
        if b_s4.button("500", use_container_width=True):
            est_foto["escala_um"] = 500.0
            st.rerun()

        val_ingresado = st.number_input(
            "O escribe otro valor (µm):",
            min_value=0.0, max_value=100000.0,
            value=float(est_foto["escala_um"]) if est_foto["escala_um"] is not None else 0.0,
            step=10.0,
            key=f"num_um_{archivo_activo.name}"
        )
        if val_ingresado > 0:
            est_foto["escala_um"] = val_ingresado

escala_visor = ANCHO_VISOR_PX / float(w_orig)
h_visor = int(round(h_orig * escala_visor))
preview_cal = img_rgb.copy()

if est_foto["cal_pts"]:
    pts_np = np.array(est_foto["cal_pts"], dtype=np.int32)
    for px, py in pts_np:
        cv2.circle(preview_cal, (px, py), max(5, int(w_orig * 0.006)), (255, 0, 0), -1)
    if es_microscopio and len(pts_np) == 2:
        cv2.line(preview_cal, tuple(pts_np[0]), tuple(pts_np[1]), (0, 255, 0), max(2, int(w_orig * 0.003)))
    elif not es_microscopio and len(pts_np) == 4:
        ord_pts = ordenar_esquinas(pts_np).astype(np.int32)
        cv2.polylines(preview_cal, [ord_pts], isClosed=True, color=(0, 255, 0), thickness=max(2, int(w_orig * 0.003)))

with c1:
    pil_cal = Image.fromarray(preview_cal).resize((ANCHO_VISOR_PX, h_visor))
    clic_cal = streamlit_image_coordinates(pil_cal, key=f"cal_{archivo_activo.name}_{len(est_foto['cal_pts'])}_{k1_lente}")

if clic_cal is not None and clic_cal != st.session_state.ultimo_clic_cal:
    st.session_state.ultimo_clic_cal = clic_cal
    if len(est_foto["cal_pts"]) < pts_req:
        est_foto["cal_pts"].append((clic_cal["x"] / escala_visor, clic_cal["y"] / escala_visor))
        st.rerun()

# Verificar si la calibración está completa para pasar al trazado
calibracion_lista = False
if es_microscopio:
    if len(est_foto["cal_pts"]) == 2:
        if est_foto["escala_um"] is not None and est_foto["escala_um"] > 0:
            calibracion_lista = True
        else:
            st.warning("⚠️ Ya marcaste los 2 extremos de la barra. Ahora selecciona o escribe su valor en **µm** (ej. 50, 100, 200) en el panel derecho para activar la medición.")
elif len(est_foto["cal_pts"]) == 4:
    calibracion_lista = True

# ==============================================================================
# PASO 2: MEDICIÓN DE ESTRUCTURAS + LIMPIEZA DE CONTRASTES FALSOS
# ==============================================================================
if calibracion_lista:
    st.divider()

    if es_microscopio:
        p_a, p_b = np.array(est_foto["cal_pts"][0]), np.array(est_foto["cal_pts"][1])
        dist_barra_px = float(np.linalg.norm(p_b - p_a))
        escala_mm = est_foto["escala_um"] / 1000.0
        px_per_mm = dist_barra_px / escala_mm
        img_trabajo_bgr = img_bgr.copy()
        img_trabajo_rgb = mejorar_contraste_microscopio(img_bgr, gamma_val, gain_val)
        mascara_macro = None
        st.success(f"📏 Escala activa en **{archivo_activo.name}**: **{dist_barra_px:.1f} px = {est_foto['escala_um']:.0f} µm** ({px_per_mm/1000.0:.3f} px/µm)")
    else:
        px_per_mm = px_per_mm_vid
        img_trabajo_bgr, img_regla = rectificar_lamina(
            img_bgr, est_foto["cal_pts"], dim1_mm, dim2_mm, px_per_mm
        )
        _, mascara_macro = segmentar_cuerpo_macro(
            img_trabajo_bgr, gamma_val, gain_val, umbral_val, px_per_mm,
            puntos_guia=st.session_state.extremos_pts,
            radio_corredor_mm=radio_corredor,
            usar_sato=usar_sato,
            puntos_borrar=est_foto["borrar_pts"]
        )
        img_trabajo_rgb = cv2.cvtColor(img_regla, cv2.COLOR_BGR2RGB)

    st.subheader(f"Paso 2: Trazado de Estructuras en [{archivo_activo.name}] → Espécimen [{id_especimen}]")

    # Filtrar estructuras disponibles según perfil de género y sexo/estadio
    estr_perfil = PERFILES_GENERO[perfil_gen]
    opciones_est = [
        c for c in CATALOGO_ESTRUCTURAS.keys()
        if c in estr_perfil and c not in st.session_state.estructuras_ausentes
    ]
    if meta_estadio != "Adulto - Hembra" and "DV" in opciones_est:
        opciones_est.remove("DV")
    if meta_estadio != "Adulto - Macho" and "LESP" in opciones_est:
        opciones_est.remove("LESP")

    col_sel_est, col_zoom, col_px, col_py = st.columns([2.4, 1.2, 1, 1])
    with col_sel_est:
        est_activa = st.selectbox(
            "🧬 Estructura anatómica a medir en esta imagen:",
            options=opciones_est,
            format_func=lambda c: f"{'✅' if c in est_foto['trazos'] else '📍'} [{c}] {CATALOGO_ESTRUCTURAS[c][0]}"
        )
    with col_zoom:
        zoom_factor = st.slider("🔍 Zoom", 1.0, 4.0, 1.0, 0.25, key=f"z_{archivo_activo.name}")
    with col_px:
        pan_x = st.slider("↔️ Pan X", 0, 100, 50, 1, disabled=(zoom_factor == 1.0), key=f"px_{archivo_activo.name}")
    with col_py:
        pan_y = st.slider("↕️ Pan Y", 0, 100, 50, 1, disabled=(zoom_factor == 1.0), key=f"py_{archivo_activo.name}")

    # Modo de interacción con el mouse: Trazar estructura vs Eliminar contraste falso
    modo_clic = "📍 Trazar puntos de estructura"
    if not es_microscopio:
        modo_clic = st.radio(
            "Acción del clic sobre la imagen:",
            ["📍 Trazar puntos de estructura", "🧹 Eliminar forma / contraste falso (clic sobre mancha)"],
            horizontal=True,
            key=f"rad_clic_{archivo_activo.name}"
        )

    pts_guia = st.session_state.extremos_pts
    overlay = img_trabajo_rgb.copy()

    if not es_microscopio and mascara_macro is not None:
        contornos, _ = cv2.findContours(mascara_macro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        verde = np.zeros_like(overlay)
        verde[mascara_macro > 0] = [0, 255, 100]
        overlay = cv2.addWeighted(overlay, 0.82, verde, 0.18, 0)
        cv2.drawContours(overlay, contornos, -1, (0, 255, 255), 2)

    # Dibujar estructuras ya fijadas en esta imagen
    for cod_g, trazo_g in est_foto["trazos"].items():
        _, color_hex, _ = CATALOGO_ESTRUCTURAS[cod_g]
        r_xy = trazo_g["ruta_xy"].astype(np.int32)
        cv2.polylines(overlay, [r_xy], isClosed=False, color=(255, 60, 60), thickness=2)
        cv2.putText(overlay, cod_g, tuple(r_xy[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

    # Dibujar puntos guía activos
    if len(pts_guia) > 1:
        cv2.polylines(overlay, [np.array(pts_guia, dtype=np.int32)], isClosed=False, color=(255, 165, 0), thickness=2)
    for idx_g, (ex, ey) in enumerate(pts_guia):
        cx, cy = int(round(ex)), int(round(ey))
        cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 0), 1)
        cv2.putText(overlay, str(idx_g + 1), (cx + 7, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

    h_trab, w_trab = overlay.shape[:2]
    crop_w = int(round(w_trab / zoom_factor))
    crop_h = int(round(h_trab / zoom_factor))
    x0 = int(round((w_trab - crop_w) * (pan_x / 100.0)))
    y0 = int(round((h_trab - crop_h) * (pan_y / 100.0)))
    overlay_crop = overlay[y0:y0 + crop_h, x0:x0 + crop_w]

    col_v1, col_v2 = st.columns([3, 1.1])
    with col_v2:
        st.metric(f"Puntos activos [{est_activa}]", f"{len(pts_guia)} (mín. 2)")

        if len(pts_guia) >= 2:
            if st.button(f"💾 Fijar Medida [{est_activa}]", type="primary", use_container_width=True):
                _, _, tipo_est = CATALOGO_ESTRUCTURAS[est_activa]
                res_trazo = medir_estructura(
                    img_trabajo_bgr, mascara_macro, pts_guia, px_per_mm,
                    tipo_estructura=tipo_est, es_microscopio=es_microscopio, suavizar_subpixel=suavizar_eje
                )
                est_foto["trazos"][est_activa] = res_trazo
                if est_activa == "LT" and not es_microscopio:
                    est_foto["top3"] = res_trazo["top3_anchos_mm"]

                st.session_state.extremos_pts = []
                st.session_state.ultimo_clic_ext = None
                st.rerun()

        b1, b2 = st.columns(2)
        with b1:
            if st.button("↩️ Deshacer", disabled=(len(pts_guia) == 0), use_container_width=True):
                st.session_state.extremos_pts.pop()
                st.session_state.ultimo_clic_ext = None
                st.rerun()
        with b2:
            if st.button("🔄 Limpiar", disabled=(len(pts_guia) == 0), use_container_width=True):
                st.session_state.extremos_pts = []
                st.session_state.ultimo_clic_ext = None
                st.rerun()

        if not es_microscopio and est_foto["borrar_pts"]:
            if st.button(f"♻️ Restaurar manchas borradas ({len(est_foto['borrar_pts'])})", use_container_width=True):
                est_foto["borrar_pts"] = []
                st.rerun()

        if est_activa in est_foto["trazos"]:
            st.divider()
            val_fijado = est_foto["trazos"][est_activa]["valor_mm"]
            st.success(f"[{est_activa}] = **{val_fijado:.3f} mm** ({val_fijado*1000:.1f} µm)")
            if st.button(f"✏️ Editar puntos de [{est_activa}]", use_container_width=True):
                trazo_prev = est_foto["trazos"].pop(est_activa)
                st.session_state.extremos_pts = list(trazo_prev["puntos_originales"])
                st.session_state.ultimo_clic_ext = None
                st.rerun()
            if st.button(f"🗑️ Borrar medida [{est_activa}]", use_container_width=True):
                est_foto["trazos"].pop(est_activa, None)
                st.rerun()

    with col_v1:
        escala_crop = ANCHO_VISOR_PX / float(crop_w)
        h_crop_visor = int(round(crop_h * escala_crop))
        pil_rect = Image.fromarray(overlay_crop).resize((ANCHO_VISOR_PX, h_crop_visor))
        clic_img = streamlit_image_coordinates(
            pil_rect,
            key=f"trazo_{archivo_activo.name}_{est_activa}_{len(pts_guia)}_{len(est_foto['borrar_pts'])}_{zoom_factor}_{pan_x}_{pan_y}_{gamma_val}_{gain_val}_{umbral_val}"
        )
        if clic_img is not None:
            coord_real = (x0 + clic_img["x"] / escala_crop, y0 + clic_img["y"] / escala_crop)
            if "Eliminar forma" in modo_clic:
                if clic_img != st.session_state.ultimo_clic_borrador:
                    st.session_state.ultimo_clic_borrador = clic_img
                    est_foto["borrar_pts"].append(coord_real)
                    st.rerun()
            else:
                if clic_img != st.session_state.ultimo_clic_ext:
                    st.session_state.ultimo_clic_ext = clic_img
                    st.session_state.extremos_pts.append(coord_real)
                    st.rerun()

# ==============================================================================
# PASO 3: CONSOLIDACIÓN MULTIFOTO DEL ESPÉCIMEN + ESTRUCTURAS AUSENTES + ÍNDICES
# ==============================================================================
st.divider()
st.subheader(f"Paso 3: Consolidado Anatómico e Índices del Espécimen [{id_especimen}]")

# Control para marcar estructuras como Ausentes (no presentes en el taxón)
with st.expander("🚫 Marcar estructuras anatómicas ausentes en este taxón (no se tomarán en cuenta)", expanded=False):
    ausentes_sel = st.multiselect(
        "Selecciona las estructuras que NO posee este espécimen:",
        options=[c for c in CATALOGO_ESTRUCTURAS.keys() if c not in ("LT", "ACM")],
        default=[c for c in CATALOGO_ESTRUCTURAS.keys() if c not in PERFILES_GENERO[perfil_gen] and c not in ("LT", "ACM")],
        format_func=lambda c: f"[{c}] {CATALOGO_ESTRUCTURAS[c][0]}"
    )
    st.session_state.estructuras_ausentes = set(ausentes_sel)

# Reunir TODAS las medidas fijadas a través de TODAS las fotos abiertas en sesión
medidas_consolidadas = {}
fotos_con_datos = []
top3_cuerpo = []

for f_obj in archivos_subidos:
    ef = obtener_estado_foto(f_obj.name)
    if ef["trazos"]:
        fotos_con_datos.append(f_obj.name)
        for cod_t, trazo_t in ef["trazos"].items():
            if cod_t not in st.session_state.estructuras_ausentes:
                medidas_consolidadas[cod_t] = round(trazo_t["valor_mm"], 3)
                if cod_t == "LT" and trazo_t["ancho_auto_mm"] > 0:
                    medidas_consolidadas["ACM"] = round(trazo_t["ancho_auto_mm"], 3)
        if ef["top3"]:
            top3_cuerpo = ef["top3"]

if medidas_consolidadas:
    cols_m = st.columns(min(6, max(2, len(medidas_consolidadas))))
    for idx_m, (k_med, v_med) in enumerate(medidas_consolidadas.items()):
        cols_m[idx_m % len(cols_m)].metric(f"{k_med}", f"{v_med:.3f} mm", f"{v_med*1000:.0f} µm")

    if top3_cuerpo:
        st.caption(
            f"📏 **Perfil de grosor continuo (Top-3 zonas más anchas):** "
            f"{', '.join([f'{w:.3f} mm' for w in top3_cuerpo])} → Máximo: **{max(top3_cuerpo):.3f} mm**"
        )

    modelos_seleccionados = st.multiselect(
        "📐 Modelos de índices taxonómicos a calcular:",
        options=[
            "Modelo Anisákidos / Estructuras Digestivas (LT/ACM, LT/LE, LT/LCI, LT/LAV, LV/LCI, LAV/LCI, LV/E...)",
            "Índices Clásicos de De Man (a, b, b', c, c', V%)"
        ],
        default=[
            "Modelo Anisákidos / Estructuras Digestivas (LT/ACM, LT/LE, LT/LCI, LT/LAV, LV/LCI, LAV/LCI, LV/E...)",
            "Índices Clásicos de De Man (a, b, b', c, c', V%)"
        ]
    )

    mod_pers, mod_deman = calcular_indices_taxonomicos(medidas_consolidadas)
    indices_mostrar = {}
    if any("Anisákidos" in m for m in modelos_seleccionados):
        indices_mostrar.update(mod_pers)
    if any("De Man" in m for m in modelos_seleccionados):
        indices_mostrar.update(mod_deman)

    if indices_mostrar:
        df_idx = pd.DataFrame([
            {"Índice": k, "Valor": (f"{v:.2f}" if v is not None else "N/A (Ausente o pendiente)")}
            for k, v in indices_mostrar.items()
        ])
        st.dataframe(df_idx.T, use_container_width=True)

    meta_dict = {
        "ID_Especimen": id_especimen,
        "Genero": meta_genero,
        "Morfotipo": meta_morfotipo,
        "Hospedador": meta_hospedador,
        "Estadio": meta_estadio,
    }

    if st.button(f"✅ Guardar Espécimen Consolidado [{id_especimen}] en la Base de Datos", type="primary"):
        # Generar ficha visual para la foto activa si tiene trazos
        if est_foto["trazos"] and calibracion_lista:
            png_bytes = generar_lamina_validacion(
                img_trabajo_rgb, est_foto["trazos"], medidas_consolidadas,
                indices_mostrar, f"{id_especimen}__{archivo_activo.name}", meta_dict, px_per_mm
            )
            st.session_state.imagenes_validadas[f"{id_especimen}__{archivo_activo.name}"] = png_bytes
            st.image(png_bytes, use_container_width=True)

        fila_consolidada = {
            "Fecha_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ID_Especimen": id_especimen,
            "Fotos_Integradas": "; ".join(fotos_con_datos),
            "Genero_Especie": meta_genero,
            "Morfotipo": meta_morfotipo,
            "Hospedador": meta_hospedador,
            "Estadio_Sexo": meta_estadio,
            "Deiridios": c_deiridios,
            "Labios": c_labios,
            "Cuticula": c_cuticula,
            "Mucron_Caudal": c_mucron,
            "Diente_Larval": c_diente,
            "Forma_Cola": c_cola,
            "Observaciones": c_obs,
        }

        for cod_m in CATALOGO_ESTRUCTURAS.keys():
            if cod_m in st.session_state.estructuras_ausentes:
                fila_consolidada[f"{cod_m}_mm"] = "Ausente"
                fila_consolidada[f"{cod_m}_um"] = "Ausente"
            else:
                val_mm = medidas_consolidadas.get(cod_m)
                fila_consolidada[f"{cod_m}_mm"] = val_mm if val_mm is not None else ""
                fila_consolidada[f"{cod_m}_um"] = round(val_mm * 1000.0, 1) if val_mm is not None else ""

        all_p, all_dm = calcular_indices_taxonomicos(medidas_consolidadas)
        for k_i, v_i in {**all_p, **all_dm}.items():
            fila_consolidada[k_i] = v_i if v_i is not None else ""

        st.session_state.base_especimenes[id_especimen] = fila_consolidada
        st.success(f"Espécimen **{id_especimen}** guardado con datos consolidados de {len(fotos_con_datos)} fotografía(s) y metadatos morfológicos.")
else:
    st.info("Fija al menos una estructura en cualquiera de las fotografías cargadas para ver el consolidado del espécimen.")

# Tabla acumulada global
if st.session_state.base_especimenes:
    st.divider()
    st.subheader("📋 Base de Datos Consolidada de Especímenes")
    st.dataframe(pd.DataFrame(list(st.session_state.base_especimenes.values())), use_container_width=True)
