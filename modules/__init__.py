from .image_processing import (
    run_detection, compute_vari, colorise_vari,
    segment_vegetation, predict_erosion_score, auto_detect_terrain,
)
from .mound_detection import (
    run_mound_pipeline, draw_mound_overlay,
    build_detection_heatmap, compute_cost_savings,
)
from .deforestation import (
    generate_vegetation_mask, remove_vegetation,
    enhance_ground_features, detect_hidden_patterns,
    build_vegetation_mask_visual,
)