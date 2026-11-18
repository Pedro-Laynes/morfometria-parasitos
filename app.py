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
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y PARÁMETROS FÍSICOS
# ==============================================================================
st.set_page_config(
    page_title="Morfometría Calibrada de Parásitos",
    page_icon="🔬",
    layout="wide"
)

ANCHO_MM = 76.2
ALTO_MM = 25.4
PX_PER_MM = 20.0   # 20 px = 1 mm -> 1 px = 50 µm
MARGEN_BORDE_PX = 30
ANCHO_VISOR_PX = 950  # Ancho fijo de visualización para escalar clics con exactitud


# ==============================================================================
# FUNCIONES DEL MOTOR DE VISIÓN POR COMPUTADORA
# ==============================================================================
def ordenar_esquinas(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def dibujar_regla_calibrada(img, w_mm, h_mm, px_per_mm):
    h, w = img.shape[:2]
    color_regla = (0, 255, 255)
    color_texto = (0, 255, 0)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color_regla, 2)

    for mm in range(0, int(np.floor(w_mm)) + 1):
        x = int(round(mm * px_per_mm))
        largo = 22 if mm % 10 == 0 else (15 if mm % 5 == 0 else 8)
        grosor = 2 if mm % 10 == 0 else 1
        if mm % 10 == 0 and mm > 0:
            cv2.putText(img, f"{mm}", (x - 10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_texto, 1, cv2.LINE_AA)
        cv2.line(img, (x, 0), (x, largo), color_regla, grosor)
        cv2.line(img, (x, h - 1), (x, h - 1 - largo), color_regla, grosor)

    for mm in range(0, int(np.floor(h_mm)) + 1):
        y = int(round(mm * px_per_mm))
        largo = 22 if mm % 10 == 0 else (15 if mm % 5 == 0 else 8)
        grosor = 2 if mm % 10 == 0 else 1
        if mm % 10 == 0 and mm > 0:
            cv2.putText(img, f"{mm}", (26, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_texto, 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (largo, y), color_regla, grosor)
        cv2.line(img, (w - 1, y), (w - 1 - largo, y), color_regla, grosor)

    return img


def rectificar_lamina(img_bgr, puntos_esquinas):
    esquinas_src = ordenar_esquinas(puntos_esquinas)
    ancho_px = np.linalg.norm(esquinas_src[1] - esquinas_src[0])
    alto_px = np.linalg.norm(esquinas_src[3] - esquinas_src[0])

    w_mm, h_mm = (ANCHO_MM, ALTO_MM) if ancho_px >= alto_px else (ALTO_MM, ANCHO_MM)
    dst_w = int(round(w_mm * PX_PER_MM))
    dst_h = int(round(h_mm * PX_PER_MM))

    esquinas_dst = np.array([
        [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(esquinas_src, esquinas_dst)
    img_rect = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC)
    img_regla = dibujar_regla_calibrada(img_rect.copy(), w_mm, h_mm, PX_PER_MM)
    return img_rect, img_regla


def segmentar_parasito(img_rect_bgr, gamma_fondo, ganancia_gusano, umbral):
    canal_L = cv2.cvtColor(img_rect_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
    L_ajustada = np.clip((canal_L ** gamma_fondo) * ganancia_gusano, 0.0, 1.0)
    img_realzada = (L_ajustada * 255).astype(np.uint8)

    mascara = (img_realzada >= umbral).astype(bool)
    mascara[:MARGEN_BORDE_PX, :] = False
    mascara[-MARGEN_BORDE_PX:, :] = False
    mascara[:, :MARGEN_BORDE_PX] = False
    mascara[:, -MARGEN_BORDE_PX:] = False

    mascara = remove_small_objects(mascara, min_size=150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_u8 = cv2.morphologyEx(mascara.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    mascara_u8 = cv2.morphologyEx(mascara_u8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_u8)
    if num_labels > 1:
        mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mascara_final = (labels == mayor).astype(np.uint8) * 255
    else:
        mascara_final = np.zeros_like(mascara_u8)

    return img_realzada, mascara_final


def calcular_morfometria_y_figura(img_rect_bgr, mascara, puntos_guia, nombre_archivo, params_seg):
    img_rgb = cv2.cvtColor(img_rect_bgr, cv2.COLOR_BGR2RGB)
    mask_bin = (mascara > 0).astype(np.uint8)

    esqueleto = skeletonize(mask_bin > 0)
    skel_ys, skel_xs = np.where(esqueleto)
    skel_coords = np.column_stack((skel_xs, skel_ys))

    if len(skel_coords) == 0:
        raise ValueError("No se detectó esqueleto. Ajusta el umbral de segmentación.")

    # Mapa de costos: prioriza el esqueleto (costo 1), permite cruzar dentro de la
    # máscara si hay cortes translúcidos (costo 25) y penaliza fuertemente salir del cuerpo (costo 500)
    costo = np.full(mask_bin.shape, 500.0, dtype=np.float64)
    costo[mask_bin > 0] = 25.0
    costo[esqueleto] = 1.0

    # Proyectar cada punto guía (waypoint) al píxel más cercano del esqueleto
    nodos_rc = []
    for pt in puntos_guia:
        pt_arr = np.array(pt, dtype=np.float32)
        idx_cercano = int(np.argmin(np.linalg.norm(skel_coords - pt_arr, axis=1)))
        nodos_rc.append((int(skel_ys[idx_cercano]), int(skel_xs[idx_cercano])))

    # Resolver la ruta geodésica tramo por tramo: P1 -> P2 -> P3 -> ... -> Pn
    tramos_rc = []
    for i in range(len(nodos_rc) - 1):
        seg_rc, _ = route_through_array(costo, nodos_rc[i], nodos_rc[i + 1], fully_connected=True)
        seg_arr = np.array(seg_rc)
        if i > 0 and len(seg_arr) > 1:
            seg_arr = seg_arr[1:]  # Evitar duplicar el nodo de unión entre tramos
        tramos_rc.append(seg_arr)

    ruta_rc = np.vstack(tramos_rc)
    ruta_xy = ruta_rc[:, ::-1]

    # Longitud Total (LT): suma de pasos del eje geodésico + micro-segmentos terminales exactos
    p_inicio_clic = np.array(puntos_guia[0], dtype=np.float32)
    p_fin_clic = np.array(puntos_guia[-1], dtype=np.float32)

    pasos_px = np.hypot(*np.diff(ruta_xy, axis=0).T) if len(ruta_xy) > 1 else np.array([0.0])
    lt_px = (
        np.sum(pasos_px)
        + np.linalg.norm(p_inicio_clic - ruta_xy[0])
        + np.linalg.norm(p_fin_clic - ruta_xy[-1])
    )
    lt_mm = float(lt_px / PX_PER_MM)
    lt_um = float(lt_mm * 1000.0)

    # Ancho Corporal Máximo (ACM) evaluado a lo largo del eje central
    mapa_radios = distance_transform_edt(mask_bin)
    margen_ext = max(1, int(len(ruta_rc) * 0.05))
    ruta_central_rc = ruta_rc[margen_ext:-margen_ext] if len(ruta_rc) > 20 else ruta_rc

    radios = mapa_radios[ruta_central_rc[:, 0], ruta_central_rc[:, 1]]
    idx_max = int(np.argmax(radios))
    radio_max_px = float(radios[idx_max])
    acm_mm = float((2.0 * radio_max_px) / PX_PER_MM)
    acm_um = float(acm_mm * 1000.0)
    ratio_lt_acm = float(lt_mm / acm_mm) if acm_mm > 0 else 0.0

    centro_acm_xy = ruta_central_rc[idx_max][::-1]
    idx_global = margen_ext + idx_max
    tangente = ruta_xy[min(len(ruta_xy) - 1, idx_global + 3)] - ruta_xy[max(0, idx_global - 3)]
    norma_t = np.linalg.norm(tangente)
    normal = np.array([-tangente[1], tangente[0]]) / norma_t if norma_t > 0 else np.array([0.0, 1.0])
    acm_p1 = centro_acm_xy + normal * radio_max_px
    acm_p2 = centro_acm_xy - normal * radio_max_px

    # Renderizar figura de validación en memoria (BytesIO)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.imshow(img_rgb)

    contornos, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contornos:
        cnt = cnt.squeeze()
        if cnt.ndim == 2:
            ax.plot(cnt[:, 0], cnt[:, 1], color="#00ffff", linewidth=1.2, alpha=0.85, label="Contorno segmentado")

    ruta_completa = np.vstack([p_inicio_clic, ruta_xy, p_fin_clic])
    ax.plot(ruta_completa[:, 0], ruta_completa[:, 1], color="#ff2a2a", linewidth=2.2, label="Eje Longitudinal Guiado (LT)")

    pts_arr = np.array(puntos_guia)
    ax.scatter([pts_arr[0, 0], pts_arr[-1, 0]], [pts_arr[0, 1], pts_arr[-1, 1]],
               color="#00ff00", edgecolors="black", s=60, zorder=6, label="Extremos (Inicio / Fin)")
    if len(pts_arr) > 2:
        ax.scatter(pts_arr[1:-1, 0], pts_arr[1:-1, 1],
                   color="#ff9900", edgecolors="black", s=45, zorder=5, label=f"Puntos guía intermedios ({len(pts_arr)-2})")

    ax.plot([acm_p1[0], acm_p2[0]], [acm_p1[1], acm_p2[1]],
            color="#ffdd00", linewidth=2.8, zorder=7, label="Ancho Corporal Máximo (ACM)")

    texto_res = (
        f"MUESTRA: {nombre_archivo}\n"
        f"LONGITUD TOTAL (LT): {lt_mm:.3f} mm ({lt_um:.0f} µm)\n"
        f"ANCHO MÁXIMO (ACM):  {acm_mm:.3f} mm ({acm_um:.0f} µm)\n"
        f"Relación LT/ACM:     {ratio_lt_acm:.2f} | Puntos guía: {len(puntos_guia)}"
    )
    ax.text(
        0.015, 0.96, texto_res, transform=ax.transAxes, verticalalignment="top",
        color="white", fontsize=10, fontweight="bold", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#111111", edgecolor="#00ff88", alpha=0.88)
    )

    ax.set_title(f"Validación Morfométrica — {nombre_archivo}", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.88)
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    registro = {
        "Fecha_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Archivo_Imagen": nombre_archivo,
        "LT_mm": round(lt_mm, 3),
        "LT_um": round(lt_um, 1),
        "ACM_mm": round(acm_mm, 3),
        "ACM_um": round(acm_um, 1),
        "Relacion_LT_ACM": round(ratio_lt_acm, 2),
        "Puntos_Guia": len(puntos_guia),
        "Escala_px_mm": PX_PER_MM,
        "Gamma": params_seg[0],
        "Gain": params_seg[1],
        "Umbral": params_seg[2],
    }
    return registro, buf.getvalue()


# ==============================================================================
# ESTADO DE SESIÓN PARA PROCESAMIENTO POR LOTES
# ==============================================================================
if "indice_actual" not in st.session_state:
    st.session_state.indice_actual = 0
if "esquinas_pts" not in st.session_state:
    st.session_state.esquinas_pts = []
if "ultimo_clic_esq" not in st.session_state:
    st.session_state.ultimo_clic_esq = None
if "extremos_pts" not in st.session_state:
    st.session_state.extremos_pts = []
if "ultimo_clic_ext" not in st.session_state:
    st.session_state.ultimo_clic_ext = None
if "calculo_confirmado" not in st.session_state:
    st.session_state.calculo_confirmado = False
if "resultados_lote" not in st.session_state:
    st.session_state.resultados_lote = {}      # {nombre_archivo: dict_metricas}
if "imagenes_validadas" not in st.session_state:
    st.session_state.imagenes_validadas = {}   # {nombre_archivo: bytes_png}


def reiniciar_puntos_foto():
    st.session_state.esquinas_pts = []
    st.session_state.ultimo_clic_esq = None
    st.session_state.extremos_pts = []
    st.session_state.ultimo_clic_ext = None
    st.session_state.calculo_confirmado = False


# ==============================================================================
# INTERFAZ DE USUARIO (STREAMLIT)
# ==============================================================================
st.title("🔬 Morfometría Calibrada de Parásitos (Lámina 76.2 × 25.4 mm)")

with st.sidebar:
    st.header("1. Carga de Lote de Fotos")
    archivos_subidos = st.file_uploader(
        "Sube una o varias fotos cenitales",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("2. Ajuste de Contraste")
    gamma_val = st.slider("Oscurecer Fondo (Gamma)", 1.0, 6.0, 4.7, 0.1)
    gain_val = st.slider("Iluminar Gusano (Gain)", 0.8, 2.5, 1.0, 0.05)
    umbral_val = st.slider("Umbral Tejido (0–255)", 50, 245, 181, 1)

    st.divider()
    st.header("3. Exportación de Resultados")
    if st.session_state.resultados_lote:
        df_res = pd.DataFrame(list(st.session_state.resultados_lote.values()))
        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📊 Descargar Tabla CSV ({len(df_res)} muestras)",
            data=csv_bytes,
            file_name="mediciones_parasitos_tesis.csv",
            mime="text/csv",
            use_container_width=True
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, img_bytes in st.session_state.imagenes_validadas.items():
                stem = fname.rsplit(".", 1)[0]
                zf.writestr(f"{stem}_medido.png", img_bytes)
        zip_buf.seek(0)

        st.download_button(
            label=f"🖼️ Descargar Imágenes ZIP ({len(st.session_state.imagenes_validadas)})",
            data=zip_buf.getvalue(),
            file_name="imagenes_validadas_tesis.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("Aún no hay mediciones guardadas en esta sesión.")

if not archivos_subidos:
    st.info("👈 Sube una o varias fotografías en el panel izquierdo para comenzar.")
    st.stop()

if st.session_state.indice_actual >= len(archivos_subidos):
    st.session_state.indice_actual = 0

# Barra de navegación entre fotos del lote
nombres_fotos = [f.name for f in archivos_subidos]
col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])

with col_nav1:
    if st.button("⬅️ Foto Anterior", disabled=(st.session_state.indice_actual == 0)):
        st.session_state.indice_actual -= 1
        reiniciar_puntos_foto()
        st.rerun()

with col_nav2:
    seleccion = st.selectbox(
        f"Muestra activa ({st.session_state.indice_actual + 1} de {len(archivos_subidos)}):",
        options=range(len(archivos_subidos)),
        index=st.session_state.indice_actual,
        format_func=lambda i: f"{'✅' if nombres_fotos[i] in st.session_state.resultados_lote else '⏳'} {nombres_fotos[i]}"
    )
    if seleccion != st.session_state.indice_actual:
        st.session_state.indice_actual = seleccion
        reiniciar_puntos_foto()
        st.rerun()

with col_nav3:
    if st.button("Foto Siguiente ➡️", disabled=(st.session_state.indice_actual >= len(archivos_subidos) - 1)):
        st.session_state.indice_actual += 1
        reiniciar_puntos_foto()
        st.rerun()

archivo_activo = archivos_subidos[st.session_state.indice_actual]
bytes_img = np.asarray(bytearray(archivo_activo.read()), dtype=np.uint8)
archivo_activo.seek(0)
img_bgr = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h_orig, w_orig = img_rgb.shape[:2]

# ==============================================================================
# PASO A: CALIBRACIÓN DE 4 ESQUINAS CON CLICS EN LA WEB
# ==============================================================================
st.subheader("Paso 1: Haz clic en las 4 esquinas del portaobjetos (76.2 × 25.4 mm)")
c1, c2 = st.columns([3, 1])
with c2:
    if st.button("🔄 Reiniciar 4 Esquinas"):
        reiniciar_puntos_foto()
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
clic_esq = streamlit_image_coordinates(pil_esq, key=f"esq_{archivo_activo.name}_{len(st.session_state.esquinas_pts)}")

if clic_esq is not None and clic_esq != st.session_state.ultimo_clic_esq:
    st.session_state.ultimo_clic_esq = clic_esq
    if len(st.session_state.esquinas_pts) < 4:
        x_real = clic_esq["x"] / escala_visor
        y_real = clic_esq["y"] / escala_visor
        st.session_state.esquinas_pts.append((x_real, y_real))
        st.rerun()

# ==============================================================================
# PASO B: RECTIFICACIÓN + SEGMENTACIÓN + TRAZADO MULTI-PUNTO GUIADO
# ==============================================================================
if len(st.session_state.esquinas_pts) == 4:
    st.divider()
    st.subheader("Paso 2: Marca la trayectoria desde la Cabeza hasta la Cola (2 o más puntos)")
    st.caption(
        "Haz clic en orden desde un extremo hasta el otro siguiendo las curvas del parásito "
        "(usa 2 puntos si es simple, o 5–10 puntos si está muy enrollado/curvado). Luego pulsa **Calcular Medición**."
    )

    img_rect, img_regla = rectificar_lamina(img_bgr, st.session_state.esquinas_pts)
    img_realzada, mascara = segmentar_parasito(img_rect, gamma_val, gain_val, umbral_val)

    # Preparar overlay con regla + contorno cian + puntos guía numerados
    overlay = cv2.cvtColor(img_regla, cv2.COLOR_BGR2RGB)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    verde = np.zeros_like(overlay)
    verde[mascara > 0] = [0, 255, 100]
    overlay = cv2.addWeighted(overlay, 0.80, verde, 0.20, 0)
    cv2.drawContours(overlay, contornos, -1, (0, 255, 255), 2)

    pts_guia = st.session_state.extremos_pts
    if len(pts_guia) > 1:
        pts_line = np.array(pts_guia, dtype=np.int32)
        cv2.polylines(overlay, [pts_line], isClosed=False, color=(255, 165, 0), thickness=2)

    for idx_g, (ex, ey) in enumerate(pts_guia):
        cx, cy = int(round(ex)), int(round(ey))
        color_pt = (0, 255, 0) if (idx_g == 0 or idx_g == len(pts_guia) - 1) else (255, 165, 0)
        cv2.circle(overlay, (cx, cy), 6, color_pt, -1)
        cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 2)
        cv2.putText(overlay, str(idx_g + 1), (cx + 8, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    col_seg1, col_seg2 = st.columns([3, 1])
    with col_seg2:
        st.metric("Puntos guía marcados", f"{len(pts_guia)} (mín. 2)")

        if len(pts_guia) >= 2:
            if st.button("📏 Calcular Medición", type="primary", use_container_width=True):
                st.session_state.calculo_confirmado = True
                st.rerun()

        b_undo, b_reset = st.columns(2)
        with b_undo:
            if st.button("↩️ Deshacer", disabled=(len(pts_guia) == 0), use_container_width=True):
                st.session_state.extremos_pts.pop()
                st.session_state.ultimo_clic_ext = None
                st.session_state.calculo_confirmado = False
                st.rerun()
        with b_reset:
            if st.button("🔄 Limpiar", disabled=(len(pts_guia) == 0), use_container_width=True):
                st.session_state.extremos_pts = []
                st.session_state.ultimo_clic_ext = None
                st.session_state.calculo_confirmado = False
                st.rerun()

        st.image(img_realzada, caption="Vista de Contraste (Canal L*)", use_container_width=True)

    with col_seg1:
        h_rect, w_rect = overlay.shape[:2]
        escala_rect = ANCHO_VISOR_PX / float(w_rect)
        h_rect_visor = int(round(h_rect * escala_rect))
        pil_rect = Image.fromarray(overlay).resize((ANCHO_VISOR_PX, h_rect_visor))

        clic_ext = streamlit_image_coordinates(
            pil_rect,
            key=f"ext_{archivo_activo.name}_{len(pts_guia)}_{gamma_val}_{gain_val}_{umbral_val}"
        )

        if clic_ext is not None and clic_ext != st.session_state.ultimo_clic_ext:
            st.session_state.ultimo_clic_ext = clic_ext
            ex_real = clic_ext["x"] / escala_rect
            ey_real = clic_ext["y"] / escala_rect
            st.session_state.extremos_pts.append((ex_real, ey_real))
            st.session_state.calculo_confirmado = False
            st.rerun()

    # ==========================================================================
    # PASO C: CÁLCULO AUTOMÁTICO Y GUARDADO EN EL LOTE
    # ==========================================================================
    if len(pts_guia) >= 2 and st.session_state.calculo_confirmado:
        st.divider()
        st.subheader("Paso 3: Resultado Morfométrico Validado")
        try:
            registro, png_bytes = calcular_morfometria_y_figura(
                img_rect, mascara, pts_guia,
                archivo_activo.name, (gamma_val, gain_val, umbral_val)
            )
            st.session_state.resultados_lote[archivo_activo.name] = registro
            st.session_state.imagenes_validadas[archivo_activo.name] = png_bytes

            m1, m2, m3 = st.columns(3)
            m1.metric("Longitud Total (LT)", f"{registro['LT_mm']:.3f} mm", f"{registro['LT_um']:.1f} µm")
            m2.metric("Ancho Corporal Máx (ACM)", f"{registro['ACM_mm']:.3f} mm", f"{registro['ACM_um']:.1f} µm")
            m3.metric("Relación LT / ACM", f"{registro['Relacion_LT_ACM']:.2f}")

            st.image(png_bytes, use_container_width=True)

            if st.session_state.indice_actual < len(archivos_subidos) - 1:
                if st.button("✅ Guardar y Pasar a la Siguiente Foto ➡️", type="primary"):
                    st.session_state.indice_actual += 1
                    reiniciar_puntos_foto()
                    st.rerun()
            else:
                st.success("🎉 ¡Has completado todas las muestras del lote! Descarga tu CSV y ZIP en el panel izquierdo.")

        except Exception as e:
            st.error(f"Error al calcular la morfometría: {e}")

# Mostrar tabla acumulada en vivo en la parte inferior
if st.session_state.resultados_lote:
    st.divider()
    st.subheader("📋 Tabla Acumulada de Mediciones del Lote")
    st.dataframe(pd.DataFrame(list(st.session_state.resultados_lote.values())), use_container_width=True)
