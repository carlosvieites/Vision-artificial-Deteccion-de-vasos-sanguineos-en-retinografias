from skimage import io, morphology, filters, exposure
import numpy as np
import matplotlib.pyplot as plt
import math

def evaluate(seg, gt):
    TP = np.logical_and(seg, gt).sum()   # Verdaderos positivos
    FP = np.logical_and(seg, ~gt).sum()  # Falsos positivos
    FN = np.logical_and(~seg, gt).sum()  # Falsos negativos
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision, recall, f1

def show_images(titles, images, max_cols=4):
    assert len(titles) == len(images), "Debe haber el mismo número de títulos e imágenes"
    n = len(images)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def method1(inputImg, disk_radius=5, sigma=0.5, threshold_factor=0.9, min_size=10):
    # 1. Canal Verde 
    green_channel = inputImg[:, :, 1]

    # 2. Máscara con erosión para eliminar el borde del ojo
    mask = green_channel > 20
    mask = morphology.binary_erosion(mask, morphology.disk(5))

    # 3. Top-hat 
    tophat = morphology.black_tophat(green_channel, morphology.disk(disk_radius))
    
    # 4. Suavizado para eliminar ruido de alta frecuencia
    tophat_smooth = filters.gaussian(tophat, sigma=sigma)

    # 5. Umbral de Otsu (con factor ajustable para mejorar resultados)
    thresh = filters.threshold_otsu(tophat_smooth)
    thresholded = tophat_smooth > (thresh * threshold_factor)

    # 6. Limpieza y máscara
    cleaned = morphology.remove_small_objects(thresholded, min_size)
    final_result = np.logical_and(cleaned, mask)

    return green_channel, tophat, tophat_smooth, thresholded, final_result


def method2(inputImg, sigmas, threshold_factor=0.5, min_size=50):
    # 1. Canal Verde y mejora de contraste (CLAHE)
    green = inputImg[:, :, 1]

    # CLAHE resalta vasos aumentando el contraste de estos con el fondo 
    green_clahe = exposure.equalize_adapthist(green)

    # 2. Máscara con erosión para eliminar el borde del ojo
    mask = green > 20
    mask = morphology.binary_erosion(mask, morphology.disk(8))
    
    # 3. Aplicar Frangi sobre la imagen para buscar formas tubulares
    vessels = filters.frangi(green_clahe, sigmas=sigmas, black_ridges=True)
    gaussian_filtered = filters.gaussian(vessels, sigma=1) 
    
    # 5. Umbral de Otsu (con factor ajustable para mejorar resultados)
    thresh = filters.threshold_otsu(gaussian_filtered)
    thresholded = gaussian_filtered > (thresh * threshold_factor)

    # 6. Limpieza y Máscara
    cleaned = morphology.remove_small_objects(thresholded, min_size=min_size)
    final_result = np.logical_and(cleaned, mask)
    
    return green, green_clahe, vessels, thresholded, final_result


def method_combined(inputImg, params1, params2):
    # 1. Método 1
    _, _, _, _, res1 = method1(inputImg, 
                               disk_radius=params1['disk_radius'], 
                               sigma=params1['sigma'], 
                               threshold_factor=params1['threshold_factor'], 
                               min_size=params1['min_size'])

    # 2. Método2
    _, _, _, _, res2 = method2(inputImg, 
                             sigmas=params2['sigmas'], 
                             threshold_factor=params2['threshold_factor'], 
                             min_size=params2['min_size'])

    # 3. UNIÓN LÓGICA (OR) Si un píxel es marcado como vaso por cualquiera de los dos métodos, se mantiene.
    combined_segmentation = np.logical_or(res1, res2)

    # 4. Limpieza final
    final_out = morphology.remove_small_objects(combined_segmentation, min_size=max(params1['min_size'], params2['min_size']))

    return res1, res2, combined_segmentation, final_out

if __name__ == "__main__":
    # Carga de la imagen
    i = 24
    inputImg = io.imread(f"MaterialVasos/{i}_training.tif")
    gt = io.imread(f"MaterialVasos/{i}_manual1.gif")  
    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = np.squeeze(gt)

    method = 2 # Cambiar entre 1, 2 y 3 (método mejorado)  

    if method == 1:
        green_channel, tophat, tophat_smooth, thresholded, outImg = method1(inputImg, disk_radius=4, sigma=1.2, threshold_factor=0.75, min_size=45)
        
        # Evaluación
        precision, recall, f1 = evaluate(thresholded, gt)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Visualización
        titles = ["Imagen original", "Ground truth", "Canal Verde", "Tophat", "Suavizado gaussiano","Umbralización con Otsu", "Eliminación de pequeños objetos"]
        images = [inputImg, gt, green_channel, tophat, tophat_smooth, thresholded, outImg]
        show_images(titles, images)

    if method == 2:
        sigmas = [0.52, 0.6, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3, 3.5, 4.5, 5, 5.5, 6]
        green, green_clahe, vessels, thresholded, outImg = method2(inputImg, sigmas, threshold_factor=0.75, min_size=45)

        # Evaluación
        precision, recall, f1 = evaluate(outImg, gt)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Visualización
        titles = ["Imagen original", "Ground truth", "Canal Verde", "Canal Verde CLAHE", "Filtro de frangi", "Umbralización con Otsu", "Eliminación de pequeños objetos"]
        images = [inputImg, gt, green, green_clahe,vessels, thresholded, outImg]
        show_images(titles, images)

    if method == 3: # Método 1 y 2 combinados
        sigmas = [0.52, 0.6, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3, 3.5, 4.5, 5, 5.5, 6]
        p1 = {'disk_radius': 4, 'sigma': 1.2, 'threshold_factor': 0.75, 'min_size': 45}
        p2_sigmas = [0.52, 0.6, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3, 3.5, 4.5, 5, 5.5, 6]
        p2 = {'sigmas': p2_sigmas, 'threshold_factor': 0.8, 'min_size': 30}

        res1, res2, combined, outImg = method_combined(inputImg, p1, p2)

        # Evaluación
        precision, recall, f1 = evaluate(outImg, gt)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Visualización
        titles = ["Imagen original", "Ground truth", "Método 1", "Método 2", "Método combinado", "Método combinado limpio"]
        images = [inputImg, gt, res1, res2, combined, outImg]
        show_images(titles, images)

    

    