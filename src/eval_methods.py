import numpy as np
import p2 as modulo_vasos
from skimage import io
import os

def calcular_metricas_promedio(image_folder, gt_folder):
    results = {
        "Metodo 1": np.zeros(3),
        "Metodo 2": np.zeros(3),
        "Metodo 3 (Combinado)": np.zeros(3)
    }

    sigmas_opt = [0.52, 0.6, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3, 3.5, 4.5, 5, 5.5, 6]
    p1 = {'disk_radius': 4, 'sigma': 1.2, 'threshold_factor': 0.75, 'min_size': 45}
    p2_params = {'sigmas': sigmas_opt, 'threshold_factor': 0.8, 'min_size': 30}

    valid_images = 0

    print(f"{'Img':<5} | {'M1 F1':<8} | {'M2 F1':<8} | {'M3 F1':<8}")
    print("-" * 40)

    for i in range(21, 29):
        img_path = os.path.join(image_folder, f"{i}_training.tif")
        gt_path = os.path.join(gt_folder, f"{i}_manual1.gif")

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue
        
        inputImg = io.imread(img_path)
        gt = io.imread(gt_path)
        
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = np.squeeze(gt)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        
        gt = (gt > 0).astype(np.bool_)

        _, _, _, _, res1 = modulo_vasos.method1(inputImg, **p1)
        m1_metrics = modulo_vasos.evaluate(res1, gt)
        results["Metodo 1"] += m1_metrics

        _, _, _, _, res2 = modulo_vasos.method2(inputImg, **p2_params)
        m2_metrics = modulo_vasos.evaluate(res2, gt)
        results["Metodo 2"] += m2_metrics

        _, _, _, res3 = modulo_vasos.method_combined(inputImg, p1, p2_params)
        m3_metrics = modulo_vasos.evaluate(res3, gt)
        results["Metodo 3 (Combinado)"] += m3_metrics

        valid_images += 1
        print(f"{i:<5} | {m1_metrics[2]:.4f} | {m2_metrics[2]:.4f} | {m3_metrics[2]:.4f}")

    print("\n" + "="*50)
    print(f"RESULTADOS FINALES PROMEDIO ({valid_images} imágenes)")
    print("="*50)
    print(f"{'Método':<20} | {'Prec.':<7} | {'Rec.':<7} | {'F1':<7}")
    print("-" * 50)

    for nombre, suma_metricas in results.items():
        media = suma_metricas / valid_images
        print(f"{nombre:<20} | {media[0]:.4f} | {media[1]:.4f} | {media[2]:.4f}")

if __name__ == "__main__":
    PATH_IMGS = "MaterialVasos/"
    PATH_GTS = "MaterialVasos/"
    calcular_metricas_promedio(PATH_IMGS, PATH_GTS)