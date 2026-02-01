import p2
import numpy as np
from skimage import io

def optimize_parameters1(image_list, gt_list, disk_range, sigma_range, factor_range, min_size_range):
    best_f1 = 0
    best_score_triple = (0, 0, 0) 
    best_params = {'disk': None, 'sigma': None, 'factor': None, 'min_size': None}

    for d in disk_range:
        for s in sigma_range:
            for f in factor_range:
                for m in min_size_range:
                    f1_scores, prec_scores, rec_scores = [], [], []

                    for img, gt in zip(image_list, gt_list):
                        _, _, _, _, thresholded = p2.method1(img, disk_radius=d, sigma=s, threshold_factor=f, min_size=m)
                        p, r, f1 = p2.evaluate(thresholded, gt)
                        prec_scores.append(p)
                        rec_scores.append(r)
                        f1_scores.append(f1)

                    mean_f1 = np.mean(f1_scores)
                    print(f"Disk: {d} | Sigma: {s:.1f} | Factor: {f:.2f} | Min size: {m} -> F1: {mean_f1:.4f}")


                    if mean_f1 > best_f1:
                        best_f1 = mean_f1
                        best_score_triple = (np.mean(prec_scores), np.mean(rec_scores), mean_f1)
                        best_params = {'disk': d, 'sigma': s, 'factor': f, 'min_size': m}

    return best_params, best_score_triple

def optimize_parameters2(image_list, gt_list, factor_range, min_size_range):
    max_f1 = 0 
    best_score_triple = (0, 0, 0) 
    best_params = {}

    sigma_configs = [
        [0.52, 0.6, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3, 3.5, 4.5, 5, 5.5, 6],                     
    ]

    for sigmas in sigma_configs:
        for f in factor_range:
            for m in min_size_range:
                f1_scores = []
                precision_scores = []
                recall_scores = []
                
                for img, gt in zip(image_list, gt_list):
 
                    _, _, _, _, cleaned = p2.method2(img, sigmas=sigmas, threshold_factor=f, min_size=m)
                    
                    prec, rec, f1 = p2.evaluate(cleaned, gt)
                    
                    f1_scores.append(f1)
                    precision_scores.append(prec)
                    recall_scores.append(rec)
            
                mean_f1 = np.mean(f1_scores)
                mean_precision = np.mean(precision_scores)
                mean_recall = np.mean(recall_scores)

                print(f"Sigmas: {sigmas} | Factor: {f:.2f} | Min size: {m} -> F1: {mean_f1:.4f}")

                if mean_f1 > max_f1:
                    max_f1 = mean_f1
                    best_score_triple = (mean_precision, mean_recall, mean_f1)
                    best_params = {'sigmas': sigmas, 'factor': f, 'min_size': m}

    return best_params, best_score_triple

def evaluate_method3(image_list, gt_list, params1, params2):
    all_precision = []
    all_recall = []
    all_f1 = []

    for img, gt in zip(image_list, gt_list):
        _, _, _, final_out = p2.method_combined(img, gt, params1, params2)

        p, r, f1 = p2.evaluate(final_out, gt)

        all_precision.append(p)
        all_recall.append(r)
        all_f1.append(f1)

    mean_p = np.mean(all_precision)
    mean_r = np.mean(all_recall)
    mean_f1 = np.mean(all_f1)

    print(f"RESULTADOS:")
    print(f"Precision media: {mean_p:.4f}")
    print(f"Recall media:    {mean_r:.4f}")
    print(f"F1-Score medio:  {mean_f1:.4f}")
    print("-" * 37)

    return mean_p, mean_r, mean_f1


if __name__ == "__main__":
    # Cargar todas las imagenes y sus ground truths
    imgs = [io.imread(f"MaterialVasos/{i}_training.tif") for i in [21, 22, 23, 24, 25, 26, 27, 28]]
    gts = [io.imread(f"MaterialVasos/{i}_manual1.gif") > 0 for i in [21, 22, 23, 24, 25, 26, 27, 28]]

    # Definir los rangos de búsqueda para método 1
    #disk_values = range(4,5)           
    #sigma_values = [1.15, 1.2, 1.25]
    #thr_factor_values = [0.7, 0.75, 0.8, 0.85] 
    #min_size_values = [ 30, 35, 40, 45, 50]

    # Definir los rangos de búsqueda para método 2
    thr_factor_values = [0.75] 
    min_size_values = [45]

    # Ejecutar optimización
    #params, score = optimize_parameters1(imgs, gts, disk_values, sigma_values, thr_factor_values, min_size_values)
    #params, score = optimize_parameters2(imgs, gts, thr_factor_values, min_size_values)

    best_p1 = {'disk_radius': 4, 'sigma': 1.2, 'threshold_factor': 0.75, 'min_size': 45}
    best_p2 = {'sigmas': [0.52, 0.6, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3, 3.5, 4.5, 5, 5.5, 6], 'threshold_factor': 0.8, 'min_size': 30}
    precision, recall, f1 = evaluate_method3(imgs, gts, best_p1, best_p2)

    #precision, recall, f1 = score
    
    print(f"F1-score medio máximo: {f1:.4f}")
    print(f"Precisión media máxima: {precision:.4f}")
    print(f"Recall medio máximo: {recall:.4f}")
    
    #print(f"MEJORES PARÁMETROS ENCONTRADOS:")

    #print(f"Disk radius: {params['disk']}")            # Para method1
    #print(f"Sigma: {params['sigma']:.1f}")             # Para method1

    #print(f"Sigmas: {params['sigmas']}")                # Para method2

    #print(f"Threshold factor: {params['factor']}")
    #print(f"Minimum size: {params['min_size']}")
    

    
    