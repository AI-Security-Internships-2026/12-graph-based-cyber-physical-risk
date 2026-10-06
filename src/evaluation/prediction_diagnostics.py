"""Calibration and model storage from actual predictions; no test-set tuning."""
from src.evaluation.robustness_stats import expected_calibration_error,brier_score

def diagnostics(probs,true,model=None):
    def arr(x):return x.detach().cpu().numpy() if hasattr(x,"detach") else x
    result={"ece":expected_calibration_error(arr(probs),arr(true)),"brier_score":brier_score(arr(probs),arr(true)),
      "false_alarms_per_hour":None,"event_recall":None,"time_to_detect_mean_s":None,
      "false_alarms_per_hour_note":"N/A: continuous exposure and timestamp units not explicitly validated",
      "event_metrics_note":"N/A: verified event IDs/onsets not supplied; heuristic grouping is not ground truth"}
    if model is not None and hasattr(model,"parameters"):
        result["model_size_bytes"]=sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
        result["model_size_definition"]="parameters and buffers; excludes framework/runtime overhead"
    else:
        result["model_size_bytes"]=None
        result["model_size_definition"]="N/A: backend-specific serialized storage not measured"
    return result
