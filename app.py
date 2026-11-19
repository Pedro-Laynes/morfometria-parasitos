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
# CONFIGURACIÓN DE PÁGINA Y ESTILOS VISUALES (CSS PERSONALIZADO)
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
    .badge-estructura {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: #1f6feb33;
        color: #58a6ff;
        border: 1px solid #1f6feb;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BIBLIOTECA DE VIDRIERÍA Y CATÁLOGO DE ESTRUCTURAS ANATÓMICAS
# ==============================================================================
BIBLIOTECA_VIDRIERIA = {
    "Portaobjetos estándar (76.2 × 25.4 mm)": (76.2, 25.4, 20.0),
    "Laminilla cubreobjetos (20.0 × 20.0 mm)": (20.0, 20.0, 40.0),
    "Laminilla cubreobjetos (22.0 × 22.0 mm)": (22.0, 22.0, 40.0),
    "Laminilla cubreobjetos (24.0 × 24.0 mm)": (24.0, 24.0, 40.0),
    "Laminilla cubreobjetos (18.0 × 18.0 mm)": (18.0, 18.0, 45.0),
    "Laminilla rectangular (24.0 × 50.0 mm)": (50.0, 24.0, 25.0),
    "Laminilla rectangular (24.0 × 60.0 mm)": (60.0, 24.0, 25.0),
    "Portaobjetos ancho (76.2 × 50.8 mm)": (76.2, 50.8, 20.0),
    "Medida personalizada...": None,
}

# Definición de estructuras medibles: (Código, Nombre completo, Color hex, Mide ancho?)
CATALOGO_ESTRUCTURAS = {
    "LT":   ("Cuerpo Completo (LT y ACM)", "#ff2a2a", True),
    "AN":   ("Anillo Nervioso a Extremo Anterior (AN)", "#00e5ff", False),
    "LE":   ("Esófago (LE y AE)", "#ff9100", True),
    "LV":   ("Ventrículo (LV y AV)", "#d500f9", True),
    "LAV":  ("Apéndice Ventricular (LAV y AAV)", "#00e676", True),
    "LCI":  ("Ciego Intestinal (LCI y ACI)", "#ffea00", True),
    "LC":   ("Cola: Ano/Cloaca a Extremo Posterior (LC y ACA)", "#2979ff", True),
    "DV":   ("Distancia Cabeza a Vulva (DV - Hembra)", "#ff4081", False),
    "LESP": ("Longitud de Espículas (LESP - Macho)", "#76ff03", False),
}

ANCHO_VISOR_PX = 950


# ==============================================================================
# FUNCIONES DE CALIBRACIÓN ÓPTICA Y HOMOGRAFÍA
# ==============================================================================
def corregir_distorsion_radial(img_bgr, k1):
    """Corrige distorsión de barril/cojín de lentes de cámara antes de rectificar."""
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


def dibujar_regla_calibrada(img, w_mm, h_mm, px_per_mm, cuadro_ref_mm=None):
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

    if cuadro_ref_mm is not None and cuadro_ref_mm > 0:
        lado_px = int(round(cuadro_ref_mm * px_per_mm))
        if lado_px < w and lado_px < h:
            cx, cy = w // 2, h // 2
            x1, y1 = cx - lado_px // 2, cy - lado_px // 2
            cv2.rectangle(img, (x1, y1), (x1 + lado_px, y1 + lado_px), (255, 180, 0), 1)
            cv2.putText(img, f"Ref. {cuadro_ref_mm:.0f}x{cuadro_ref_mm:.0f} mm", (x1 + 6, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)
    return img


def rectificar_lamina(img_bgr, puntos_esquinas, dim1_mm, dim2_mm, px_per_mm, cuadro_ref_mm=None):
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
    img_regla = dibujar_regla_calibrada(img_rect.copy(), w_mm, h_mm, px_per_mm, cuadro_ref_mm)
    return img_rect, img_regla


# ==============================================================================
# VISIÓN POR COMPUTADORA: REALCE TUBULAR (SATO) Y SEGMENTACIÓN GUIADA
# ==============================================================================
def crear_corredor_roi(shape_hw, puntos_guia, px_per_mm, radio_mm=0.8):
    h, w = shape_hw
    roi = np.zeros((h, w), dtype=np.uint8)
    if len(puntos_guia) < 2:
        roi[:, :] = 255
        return roi
    pts = np.array(puntos_guia, dtype=np.int32)
    grosor_px = max(8, int(round(2.0 * radio_mm * px_per_mm)))
    cv2.polylines(roi, [pts], isClosed=False, color=255, thickness=grosor_px)
    return roi


def segmentar_estructura(img_rect_bgr, gamma_fondo, ganancia_gusano, umbral, px_per_mm,
                         puntos_guia=None, radio_corredor_mm=0.8, usar_sato=False):
    canal_L = cv2.cvtColor(img_rect_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
    L_u8 = (canal_L * 255).astype(np.uint8)

    k_size = max(15, int(round(1.5 * px_per_mm)) | 1)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(L_u8, cv2.MORPH_TOPHAT, kernel_tophat).astype(np.float32) / 255.0

    base_L = np.clip((canal_L ** gamma_fondo) * ganancia_gusano + 0.40 * tophat, 0.0, 1.0)

    if usar_sato:
        # Filtro Hessiano de Sato para realzar estructuras tubulares/filamentosas translúcidas
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

    tiene_guia = puntos_guia is not None and len(puntos_guia) >= 2
    if tiene_guia:
        roi_tubo = crear_corredor_roi(mascara.shape, puntos_guia, px_per_mm, radio_mm=radio_corredor_mm)
        mascara = mascara & (roi_tubo > 0)

    mascara = remove_small_objects(mascara, min_size=max(15, int(0.08 * px_per_mm ** 2)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_u8 = cv2.morphologyEx(mascara.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    mascara_u8 = cv2.morphologyEx(mascara_u8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_u8)
    if num_labels > 1:
        if tiene_guia:
            mascara_final = mascara_u8
        else:
            mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mascara_final = (labels == mayor).astype(np.uint8) * 255
    else:
        mascara_final = np.zeros_like(mascara_u8)

    return img_realzada, mascara_final


def suavizar_curva_subpixel(ruta_xy, n_puntos=300):
    """Refinamiento subpíxel mediante spline cúbico suavizado para evitar zigzags discretos."""
    pts = np.array(ruta_xy, dtype=np.float64)
    if len(pts) < 4:
        return pts
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = np.vstack([pts[0], pts[1:][diffs > 0.2]])
    if len(pts) < 4:
        return pts
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 0.35, k=3)
        u_fine = np.linspace(0, 1, n_puntos)
        xs, ys = splev(u_fine, tck)
        curva = np.column_stack((xs, ys))
        curva[0] = pts[0]
        curva[-1] = pts[-1]
        return curva
    except Exception:
        return pts


def medir_trazo_estructura(img_rect_bgr, mascara, puntos_guia, px_per_mm,
                           medir_ancho=True, suavizar_subpixel=True):
    """
    Calcula la longitud geodésica de una estructura y, si corresponde, evalúa el perfil
    de grosor continuo extrayendo los 3 picos más anchos para seleccionar el máximo real.
    """
    mask_bin = (mascara > 0).astype(np.uint8)
    h_img, w_img = mask_bin.shape

    esqueleto = skeletonize(mask_bin > 0)
    skel_ys, skel_xs = np.where(esqueleto)
    skel_coords = np.column_stack((skel_xs, skel_ys))

    p_inicio = np.array(puntos_guia[0], dtype=np.float64)
    p_fin = np.array(puntos_guia[-1], dtype=np.float64)

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

        ruta_rc = np.vstack(tramos_rc)
        ruta_xy = ruta_rc[:, ::-1].astype(np.float64)
    else:
        # Interpolación directa entre puntos guía si el órgano interno es muy tenue
        segmentos = []
        pts_arr = np.array(puntos_guia, dtype=np.float64)
        for i in range(len(pts_arr) - 1):
            t = np.linspace(0, 1, 80, endpoint=(i == len(pts_arr) - 2))
            segmentos.append(pts_arr[i] + np.outer(t, pts_arr[i + 1] - pts_arr[i]))
        ruta_xy = np.vstack(segmentos)

    if len(ruta_xy) > 4:
        if np.linalg.norm(p_inicio - ruta_xy[0]) > 1.5:
            ruta_xy = np.vstack([p_inicio, ruta_xy[2:]])
        if np.linalg.norm(p_fin - ruta_xy[-1]) > 1.5:
            ruta_xy = np.vstack([ruta_xy[:-2], p_fin])

    if suavizar_subpixel:
        ruta_xy = suavizar_curva_subpixel(ruta_xy)

    pasos_px = np.hypot(*np.diff(ruta_xy, axis=0).T) if len(ruta_xy) > 1 else np.array([0.0])
    longitud_mm = float(np.sum(pasos_px) / px_per_mm)

    # Perfil de grosor continuo (3 zonas más anchas)
    ancho_max_mm = 0.0
    top3_anchos_mm = []
    segmento_ancho = None

    if medir_ancho and np.sum(mask_bin) > 15:
        mapa_radios = distance_transform_edt(mask_bin)
        ruta_rc_eval = np.column_stack((
            np.clip(np.round(ruta_xy[:, 1]).astype(int), 0, h_img - 1),
            np.clip(np.round(ruta_xy[:, 0]).astype(int), 0, w_img - 1)
        ))
        margen_ext = max(1, int(len(ruta_rc_eval) * 0.05))
        central_rc = ruta_rc_eval[margen_ext:-margen_ext] if len(ruta_rc_eval) > 20 else ruta_rc_eval
        radios = mapa_radios[central_rc[:, 0], central_rc[:, 1]]

        if len(radios) > 0:
            # Detectar los 3 picos locales principales de grosor separados al menos 10% del largo
            dist_min_picos = max(1, len(radios) // 8)
            picos, _ = find_peaks(radios, distance=dist_min_picos)
            if len(picos) == 0:
                picos = np.array([int(np.argmax(radios))])
            picos_ordenados = picos[np.argsort(radios[picos])[::-1]][:3]
            top3_anchos_mm = [round(float(2.0 * radios[p] / px_per_mm), 3) for p in picos_ordenados]

            idx_max = int(picos_ordenados[0])
            radio_max_px = float(radios[idx_max])
            ancho_max_mm = float((2.0 * radio_max_px) / px_per_mm)

            centro_xy = central_rc[idx_max][::-1]
            idx_glob = margen_ext + idx_max
            tangente = ruta_xy[min(len(ruta_xy) - 1, idx_glob + 3)] - ruta_xy[max(0, idx_glob - 3)]
            norma_t = np.linalg.norm(tangente)
            normal = np.array([-tangente[1], tangente[0]]) / norma_t if norma_t > 0 else np.array([0.0, 1.0])
            segmento_ancho = (centro_xy + normal * radio_max_px, centro_xy - normal * radio_max_px)

    return {
        "longitud_mm": longitud_mm,
        "ancho_mm": ancho_max_mm,
        "top3_anchos_mm": top3_anchos_mm,
        "ruta_xy": ruta_xy,
        "segmento_ancho": segmento_ancho,
        "mascara": mask_bin,
    }


def calcular_indices_taxonomicos(medidas):
    """
    Calcula tanto el Modelo Personalizado (Anisákidos / Estructuras Digestivas)
    como los Índices Clásicos de De Man a partir de las estructuras medidas.
    """
    def ratio(num_key, den_key, factor=1.0):
        num = medidas.get(num_key)
        den = medidas.get(den_key)
        if num is not None and den is not None and den > 0:
            return round((num / den) * factor, 2)
        return None

    # 1. Modelo Personalizado (Anisákidos / Estructuras Digestivas)
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

    # 2. Índices Clásicos de De Man
    le = medidas.get("LE")
    lv = medidas.get("LV")
    lt = medidas.get("LT")
    le_total_gland = (le + lv) if (le is not None and lv is not None) else None
    b_prima = round(lt / le_total_gland, 2) if (lt and le_total_gland and le_total_gland > 0) else None

    indice_deman = {
        "DeMan_a (LT/ACM)": ratio("LT", "ACM"),
        "DeMan_b (LT/LE)": ratio("LT", "LE"),
        "DeMan_b' (LT/[LE+LV])": b_prima,
        "DeMan_c (LT/LC)": ratio("LT", "LC"),
        "DeMan_c' (LC/ACA)": ratio("LC", "ACA"),
        "DeMan_V% (DV/LT*100)": ratio("DV", "LT", 100.0),
    }

    return modelo_personalizado, indice_deman


def generar_lamina_validacion(img_rect_bgr, trazos_guardados, medidas_dict, indices_activos,
                              etiqueta_muestra, meta_dict, px_per_mm):
    img_rgb = cv2.cvtColor(img_rect_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(img_rgb)

    for cod, trazo in trazos_guardados.items():
        nombre_est, color_hex, _ = CATALOGO_ESTRUCTURAS[cod]
        ruta = trazo["ruta_xy"]
        ax.plot(ruta[:, 0], ruta[:, 1], color=color_hex, linewidth=2.1, label=f"{cod}: {trazo['longitud_mm']:.3f} mm")
        ax.scatter([ruta[0, 0], ruta[-1, 0]], [ruta[0, 1], ruta[-1, 1]],
                   color=color_hex, edgecolors="black", s=42, zorder=6)

        if trazo["segmento_ancho"] is not None and trazo["ancho_mm"] > 0:
            p1, p2 = trazo["segmento_ancho"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ffffff", linestyle="--", linewidth=2.0, zorder=7)

    # Cuadro resumen superior izquierdo
    lineas_texto = [
        f"MUESTRA: {etiqueta_muestra}",
        f"TAXÓN: {meta_dict['Genero']} | {meta_dict['Estadio']} | Hosp: {meta_dict['Hospedador']}",
        f"LT: {medidas_dict.get('LT', 0):.3f} mm ({medidas_dict.get('LT', 0)*1000:.0f} µm) | ACM: {medidas_dict.get('ACM', 0):.3f} mm",
    ]
    resumen_idx = [f"{k}={v}" for k, v in indices_activos.items() if v is not None][:6]
    if resumen_idx:
        lineas_texto.append("ÍNDICES: " + " | ".join(resumen_idx))

    ax.text(
        0.015, 0.96, "\n".join(lineas_texto), transform=ax.transAxes, verticalalignment="top",
        color="white", fontsize=9, fontweight="bold", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#111111", edgecolor="#00ff88", alpha=0.88)
    )

    ax.set_title(f"Ficha Morfométrica Taxonómica — {etiqueta_muestra}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.88, fontsize=8.5)
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# ESTADO DE SESIÓN
# ==============================================================================
defaults = {
    "indice_actual": 0,
    "esquinas_pts": [],
    "ultimo_clic_esq": None,
    "extremos_pts": [],
    "ultimo_clic_ext": None,
    "contador_parasito_foto": 1,
    "trazos_parasito_actual": {},   # {cod_estructura: dict_trazo}
    "medidas_parasito_actual": {},  # {LT: float, ACM: float, AN: float, ...}
    "top3_acm_actual": [],
    "resultados_lote": {},
    "imagenes_validadas": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reiniciar_todo_foto():
    st.session_state.esquinas_pts = []
    st.session_state.ultimo_clic_esq = None
    st.session_state.extremos_pts = []
    st.session_state.ultimo_clic_ext = None
    st.session_state.contador_parasito_foto = 1
    st.session_state.trazos_parasito_actual = {}
    st.session_state.medidas_parasito_actual = {}
    st.session_state.top3_acm_actual = []


def siguiente_parasito_misma_foto():
    st.session_state.extremos_pts = []
    st.session_state.ultimo_clic_ext = None
    st.session_state.contador_parasito_foto += 1
    st.session_state.trazos_parasito_actual = {}
    st.session_state.medidas_parasito_actual = {}
    st.session_state.top3_acm_actual = []


# ==============================================================================
# BARRA LATERAL: LOTE, METADATOS, VIDRIERÍA Y ÓPTICA
# ==============================================================================
st.title("🔬 Sistema Integrado de Morfometría Taxonómica de Nematodos")

with st.sidebar:
    st.header("1. Carga de Lote de Fotos")
    archivos_subidos = st.file_uploader(
        "Sube fotografías cenitales",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("2. Metadatos Taxonómicos")
    meta_genero = st.text_input("Género / Especie", value="Anisakis sp.")
    meta_morfotipo = st.selectbox("Morfotipo", ["Tipo I", "Tipo II", "Tipo III", "Morfotipo A", "Morfotipo B", "No aplica"])
    meta_hospedador = st.text_input("Hospedador", value="No especificado")
    meta_estadio = st.selectbox(
        "Estadio / Sexo",
        ["Larva (L3/L4)", "Adulto - Hembra", "Adulto - Macho"]
    )

    st.divider()
    st.header("3. Referencia Física (Vidriería)")
    opcion_ref = st.selectbox("Objeto calibrado (4 esquinas):", list(BIBLIOTECA_VIDRIERIA.keys()), index=0)
    if BIBLIOTECA_VIDRIERIA[opcion_ref] is not None:
        dim1_mm, dim2_mm, px_per_mm = BIBLIOTECA_VIDRIERIA[opcion_ref]
    else:
        c_d1, c_d2 = st.columns(2)
        dim1_mm = c_d1.number_input("Ancho (mm)", 1.0, 200.0, 22.0, 0.5)
        dim2_mm = c_d2.number_input("Alto (mm)", 1.0, 200.0, 22.0, 0.5)
        px_per_mm = 40.0 if max(dim1_mm, dim2_mm) <= 30.0 else 20.0

    cuadro_ref_mm = None
    if "Portaobjetos" in opcion_ref:
        refuerzo = st.selectbox("Retícula de verificación:", ["Ninguna", "Laminilla 20 × 20 mm", "Laminilla 22 × 22 mm", "Laminilla 24 × 24 mm"])
        if "20" in refuerzo:
            cuadro_ref_mm = 20.0
        elif "22" in refuerzo:
            cuadro_ref_mm = 22.0
        elif "24" in refuerzo:
            cuadro_ref_mm = 24.0

    st.divider()
    st.header("4. Contraste y Visión Avanzada")
    gamma_val = st.slider("Oscurecer Fondo (Gamma)", 1.0, 6.0, 2.5, 0.1)
    gain_val = st.slider("Iluminar Espécimen (Gain)", 0.8, 3.0, 1.4, 0.05)
    umbral_val = st.slider("Umbral Tejido (0–255)", 20, 245, 120, 1)
    radio_corredor = st.slider("Radio de búsqueda (mm)", 0.15, 2.5, 0.6, 0.05)

    with st.expander("🛠️ Visión Avanzada (Filtros Tubulares y Óptica)", expanded=False):
        usar_sato = st.checkbox("Activar Filtro Tubular de Sato (nematodos muy translúcidos)", value=False)
        suavizar_eje = st.checkbox("Refinamiento Subpíxel de Curva (Spline)", value=True)
        k1_lente = st.slider("Corrección Distorsión Radial Lente (k1)", -0.30, 0.30, 0.0, 0.02)

    st.divider()
    st.header("5. Exportación de Resultados")
    if st.session_state.resultados_lote:
        df_res = pd.DataFrame(list(st.session_state.resultados_lote.values()))
        st.download_button(
            label=f"📊 Descargar Base de Datos CSV ({len(df_res)} muestras)",
            data=df_res.to_csv(index=False).encode("utf-8"),
            file_name="morfometria_taxonomica_tesis.csv",
            mime="text/csv",
            use_container_width=True
        )
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for id_m, img_bytes in st.session_state.imagenes_validadas.items():
                limpio = id_m.replace(" ", "_").replace("#", "P")
                zf.writestr(f"{limpio}_ficha.png", img_bytes)
        zip_buf.seek(0)
        st.download_button(
            label=f"🖼️ Descargar Fichas ZIP ({len(st.session_state.imagenes_validadas)})",
            data=zip_buf.getvalue(),
            file_name="fichas_morfometricas_tesis.zip",
            mime="application/zip",
            use_container_width=True
        )

if not archivos_subidos:
    st.info("👈 Sube una o varias fotografías y configura los metadatos taxonómicos en el panel izquierdo.")
    st.stop()

if st.session_state.indice_actual >= len(archivos_subidos):
    st.session_state.indice_actual = 0

# Navegación de fotos
nombres_fotos = [f.name for f in archivos_subidos]
c_n1, c_n2, c_n3 = st.columns([1, 3, 1])
with c_n1:
    if st.button("⬅️ Foto Anterior", disabled=(st.session_state.indice_actual == 0)):
        st.session_state.indice_actual -= 1
        reiniciar_todo_foto()
        st.rerun()
with c_n2:
    sel = st.selectbox(
        f"Fotografía activa ({st.session_state.indice_actual + 1} de {len(archivos_subidos)}):",
        options=range(len(archivos_subidos)),
        index=st.session_state.indice_actual,
        format_func=lambda i: f"{'✅' if any(k.startswith(nombres_fotos[i]) for k in st.session_state.resultados_lote) else '⏳'} {nombres_fotos[i]}"
    )
    if sel != st.session_state.indice_actual:
        st.session_state.indice_actual = sel
        reiniciar_todo_foto()
        st.rerun()
with c_n3:
    if st.button("Foto Siguiente ➡️", disabled=(st.session_state.indice_actual >= len(archivos_subidos) - 1)):
        st.session_state.indice_actual += 1
        reiniciar_todo_foto()
        st.rerun()

archivo_activo = archivos_subidos[st.session_state.indice_actual]
bytes_img = np.asarray(bytearray(archivo_activo.read()), dtype=np.uint8)
archivo_activo.seek(0)
img_bgr_raw = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
img_bgr = corregir_distorsion_radial(img_bgr_raw, k1_lente)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h_orig, w_orig = img_rgb.shape[:2]

# ==============================================================================
# PASO 1: CALIBRACIÓN DE 4 ESQUINAS
# ==============================================================================
st.subheader(f"Paso 1: Calibración de 4 esquinas ({dim1_mm} × {dim2_mm} mm)")
c1, c2 = st.columns([3, 1])
with c2:
    if st.button("🔄 Reiniciar Calibración"):
        reiniciar_todo_foto()
        st.rerun()
    st.metric("Esquinas marcadas", f"{len(st.session_state.esquinas_pts)} / 4")

escala_visor = ANCHO_VISOR_PX / float(w_orig)
h_visor = int(round(h_orig * escala_visor))
preview_esq = img_rgb.copy()

if st.session_state.esquinas_pts:
    pts_np = np.array(st.session_state.esquinas_pts, dtype=np.int32)
    for px, py in pts_np:
        cv2.circle(preview_esq, (px, py), max(6, int(w_orig * 0.008)), (255, 0, 0), -1)
    if len(pts_np) == 4:
        ord_pts = ordenar_esquinas(pts_np).astype(np.int32)
        cv2.polylines(preview_esq, [ord_pts], isClosed=True, color=(0, 255, 0), thickness=max(2, int(w_orig * 0.003)))

pil_esq = Image.fromarray(preview_esq).resize((ANCHO_VISOR_PX, h_visor))
clic_esq = streamlit_image_coordinates(pil_esq, key=f"esq_{archivo_activo.name}_{len(st.session_state.esquinas_pts)}_{k1_lente}")

if clic_esq is not None and clic_esq != st.session_state.ultimo_clic_esq:
    st.session_state.ultimo_clic_esq = clic_esq
    if len(st.session_state.esquinas_pts) < 4:
        st.session_state.esquinas_pts.append((clic_esq["x"] / escala_visor, clic_esq["y"] / escala_visor))
        st.rerun()

# ==============================================================================
# PASO 2: MEDICIÓN MULTI-ESTRUCTURA (LT, ACM, AN, LE/AE, LV/AV, LAV/AAV, LCI/ACI...)
# ==============================================================================
if len(st.session_state.esquinas_pts) == 4:
    st.divider()
    etiqueta_actual = f"{archivo_activo.name} #P{st.session_state.contador_parasito_foto}"
    st.subheader(f"Paso 2: Trazado de Estructuras Anatómicas — {etiqueta_actual}")

    # Filtrar estructuras según estadio/sexo seleccionado en metadatos
    opciones_est = ["LT", "AN", "LE", "LV", "LAV", "LCI", "LC"]
    if meta_estadio == "Adulto - Hembra":
        opciones_est.append("DV")
    elif meta_estadio == "Adulto - Macho":
        opciones_est.append("LESP")

    col_sel_est, col_zoom, col_px, col_py = st.columns([2.2, 1.2, 1, 1])
    with col_sel_est:
        est_activa = st.selectbox(
            "🧬 Selecciona la estructura a trazar ahora:",
            options=opciones_est,
            format_func=lambda c: f"{'✅' if c in st.session_state.trazos_parasito_actual else '📍'} [{c}] {CATALOGO_ESTRUCTURAS[c][0]}"
        )
    with col_zoom:
        zoom_factor = st.slider("🔍 Zoom", 1.0, 4.0, 1.0, 0.25)
    with col_px:
        pan_x = st.slider("↔️ Pan X", 0, 100, 50, 1, disabled=(zoom_factor == 1.0))
    with col_py:
        pan_y = st.slider("↕️ Pan Y", 0, 100, 50, 1, disabled=(zoom_factor == 1.0))

    pts_guia = st.session_state.extremos_pts
    img_rect, img_regla = rectificar_lamina(
        img_bgr, st.session_state.esquinas_pts, dim1_mm, dim2_mm, px_per_mm, cuadro_ref_mm
    )
    img_realzada, mascara = segmentar_estructura(
        img_rect, gamma_val, gain_val, umbral_val, px_per_mm,
        puntos_guia=pts_guia, radio_corredor_mm=radio_corredor, usar_sato=usar_sato
    )

    overlay = cv2.cvtColor(img_regla, cv2.COLOR_BGR2RGB)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    verde = np.zeros_like(overlay)
    verde[mascara > 0] = [0, 255, 100]
    overlay = cv2.addWeighted(overlay, 0.80, verde, 0.20, 0)
    cv2.drawContours(overlay, contornos, -1, (0, 255, 255), 2)

    # Dibujar trazos ya confirmados de otras estructuras
    for cod_g, trazo_g in st.session_state.trazos_parasito_actual.items():
        r_xy = trazo_g["ruta_xy"].astype(np.int32)
        cv2.polylines(overlay, [r_xy], isClosed=False, color=(255, 50, 50), thickness=2)

    # Dibujar puntos guía activos
    if len(pts_guia) > 1:
        cv2.polylines(overlay, [np.array(pts_guia, dtype=np.int32)], isClosed=False, color=(255, 165, 0), thickness=2)
    for idx_g, (ex, ey) in enumerate(pts_guia):
        cx, cy = int(round(ex)), int(round(ey))
        cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(overlay, str(idx_g + 1), (cx + 7, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

    h_rect, w_rect = overlay.shape[:2]
    crop_w = int(round(w_rect / zoom_factor))
    crop_h = int(round(h_rect / zoom_factor))
    x0 = int(round((w_rect - crop_w) * (pan_x / 100.0)))
    y0 = int(round((h_rect - crop_h) * (pan_y / 100.0)))

    overlay_crop = overlay[y0:y0 + crop_h, x0:x0 + crop_w]
    realzada_crop = img_realzada[y0:y0 + crop_h, x0:x0 + crop_w]

    col_v1, col_v2 = st.columns([3, 1])
    with col_v2:
        st.metric(f"Puntos en [{est_activa}]", f"{len(pts_guia)} (mín. 2)")
        if len(pts_guia) >= 2:
            if st.button(f"💾 Fijar Medida [{est_activa}]", type="primary", use_container_width=True):
                _, _, mide_ancho = CATALOGO_ESTRUCTURAS[est_activa]
                res_trazo = medir_trazo_estructura(
                    img_rect, mascara, pts_guia, px_per_mm,
                    medir_ancho=mide_ancho, suavizar_subpixel=suavizar_eje
                )
                st.session_state.trazos_parasito_actual[est_activa] = res_trazo
                st.session_state.medidas_parasito_actual[est_activa] = round(res_trazo["longitud_mm"], 3)

                # Asignar automáticamente anchos correspondientes
                mapa_anchos = {"LT": "ACM", "LE": "AE", "LV": "AV", "LAV": "AAV", "LCI": "ACI", "LC": "ACA"}
                if est_activa in mapa_anchos and res_trazo["ancho_mm"] > 0:
                    st.session_state.medidas_parasito_actual[mapa_anchos[est_activa]] = round(res_trazo["ancho_mm"], 3)
                if est_activa == "LT":
                    st.session_state.top3_acm_actual = res_trazo["top3_anchos_mm"]

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

        st.image(realzada_crop, caption="Vista Procesada (L* / Sato)", use_container_width=True)

    with col_v1:
        escala_crop = ANCHO_VISOR_PX / float(crop_w)
        h_crop_visor = int(round(crop_h * escala_crop))
        pil_rect = Image.fromarray(overlay_crop).resize((ANCHO_VISOR_PX, h_crop_visor))
        clic_ext = streamlit_image_coordinates(
            pil_rect,
            key=f"trazo_{etiqueta_actual}_{est_activa}_{len(pts_guia)}_{zoom_factor}_{pan_x}_{pan_y}_{gamma_val}_{gain_val}_{umbral_val}_{usar_sato}"
        )
        if clic_ext is not None and clic_ext != st.session_state.ultimo_clic_ext:
            st.session_state.ultimo_clic_ext = clic_ext
            st.session_state.extremos_pts.append((x0 + clic_ext["x"] / escala_crop, y0 + clic_ext["y"] / escala_crop))
            st.rerun()

    # ==========================================================================
    # PASO 3: MEDIDAS CONCRETAS, PERFIL TOP-3 ANCHOS Y SELECTOR DE ÍNDICES
    # ==========================================================================
    if st.session_state.medidas_parasito_actual:
        st.divider()
        st.subheader(f"Paso 3: Medidas Concretas e Índices Taxonómicos — {etiqueta_actual}")

        med = st.session_state.medidas_parasito_actual
        cols_m = st.columns(min(6, max(2, len(med))))
        for idx_m, (k_med, v_med) in enumerate(med.items()):
            cols_m[idx_m % len(cols_m)].metric(f"{k_med}", f"{v_med:.3f} mm", f"{v_med*1000:.0f} µm")

        if st.session_state.top3_acm_actual:
            st.caption(
                f"📏 **Perfil de grosor continuo (Top-3 zonas más anchas del cuerpo):** "
                f"{', '.join([f'{w:.3f} mm' for w in st.session_state.top3_acm_actual])} "
                f"→ Seleccionado máximo: **{max(st.session_state.top3_acm_actual):.3f} mm**"
            )

        # Selector desplegable de Modelos de Índices
        modelos_seleccionados = st.multiselect(
            "📐 Selecciona los modelos de índices taxonómicos a calcular y presentar:",
            options=[
                "Modelo Anisákidos / Estructuras Digestivas (LT/ACM, LT/LE, LT/LCI, LT/LAV, LV/LCI, LAV/LCI, LV/E...)",
                "Índices Clásicos de De Man (a, b, b', c, c', V%)"
            ],
            default=[
                "Modelo Anisákidos / Estructuras Digestivas (LT/ACM, LT/LE, LT/LCI, LT/LAV, LV/LCI, LAV/LCI, LV/E...)",
                "Índices Clásicos de De Man (a, b, b', c, c', V%)"
            ]
        )

        mod_pers, mod_deman = calcular_indices_taxonomicos(med)
        indices_mostrar = {}
        if any("Anisákidos" in m for m in modelos_seleccionados):
            indices_mostrar.update(mod_pers)
        if any("De Man" in m for m in modelos_seleccionados):
            indices_mostrar.update(mod_deman)

        if indices_mostrar:
            df_idx = pd.DataFrame([
                {"Índice": k, "Valor": (f"{v:.2f}" if v is not None else "Pendiente (trazar estructura)")}
                for k, v in indices_mostrar.items()
            ])
            st.dataframe(df_idx.T, use_container_width=True)

        meta_dict = {
            "Genero": meta_genero,
            "Morfotipo": meta_morfotipo,
            "Hospedador": meta_hospedador,
            "Estadio": meta_estadio,
        }

        if st.button("✅ Consolidar Parásito en la Base de Datos y Generar Ficha", type="primary"):
            png_bytes = generar_lamina_validacion(
                img_rect, st.session_state.trazos_parasito_actual, med,
                indices_mostrar, etiqueta_actual, meta_dict, px_per_mm
            )
            fila_registro = {
                "Fecha_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ID_Muestra": etiqueta_actual,
                "Genero_Especie": meta_genero,
                "Morfotipo": meta_morfotipo,
                "Hospedador": meta_hospedador,
                "Estadio_Sexo": meta_estadio,
                "Referencia_Calibracion": opcion_ref,
                "Escala_px_mm": px_per_mm,
            }
            # Añadir todas las medidas concretas en mm y µm
            for cod_m in ["LT", "ACM", "AN", "LE", "AE", "LCI", "ACI", "LV", "AV", "LAV", "AAV", "LC", "ACA", "DV", "LESP"]:
                val_mm = med.get(cod_m)
                fila_registro[f"{cod_m}_mm"] = val_mm if val_mm is not None else ""
                fila_registro[f"{cod_m}_um"] = round(val_mm * 1000.0, 1) if val_mm is not None else ""

            # Añadir todos los índices calculados
            for k_i, v_i in indices_mostrar.items():
                fila_registro[k_i] = v_i if v_i is not None else ""

            st.session_state.resultados_lote[etiqueta_actual] = fila_registro
            st.session_state.imagenes_validadas[etiqueta_actual] = png_bytes
            st.image(png_bytes, use_container_width=True)
            st.success(f"Muestra **{etiqueta_actual}** guardada en el lote.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("➕ Medir otro parásito en esta misma laminilla", use_container_width=True):
                siguiente_parasito_misma_foto()
                st.rerun()
        with col_b2:
            if st.session_state.indice_actual < len(archivos_subidos) - 1:
                if st.button("➡️ Pasar a la Siguiente Fotografía", use_container_width=True):
                    st.session_state.indice_actual += 1
                    reiniciar_todo_foto()
                    st.rerun()

# Tabla acumulada global
if st.session_state.resultados_lote:
    st.divider()
    st.subheader("📋 Base de Datos Acumulada de la Sesión")
    st.dataframe(pd.DataFrame(list(st.session_state.resultados_lote.values())), use_container_width=True)
