# pip install opencv-python numpy pandas ultralytics 설치 필요

import cv2
import numpy as np
import pandas as pd
import json
import os
import time  
from datetime import datetime
from ultralytics import YOLO

# -------------------------------------------------------------
# [설정 영역] 실험에 사용할 영상 파일 이름을 적어주세요
video_path = "C021_A19_SY17_P07_S07_01DAS.mp4"         
# -------------------------------------------------------------

# 1. 최고사양 통제군 결과를 격리 보관할 전용 루트 폴더 정의
root_dir = "experiment_outputs_ultra"
os.makedirs(root_dir, exist_ok=True)

# 2. 이번 실험을 위한 고유 하위 폴더명 생성 (초 단위 타임스탬프 적용)
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_dir_name = f"experiment_ultra_{time_stamp}"

# 3. 실험용 하위 폴더 경로 생성
output_dir = os.path.join(root_dir, sub_dir_name)
os.makedirs(output_dir, exist_ok=True)

# --- [울트라 최고 사양 정밀 추론 및 채점 엔진] ---
# 가장 거대하고 정확한 최상위 등급 모델 로드
model = YOLO('yolov8x-seg.pt')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"동영상 파일을 열 수 없습니다: {video_path}")
    exit()

origin_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
origin_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps > 60:
    fps = 30.0

# 하위 폴더 경로 밑으로 비디오 파일 저장 세팅
output_blur_path = os.path.join(output_dir, f"M_blur_ultra_{time_stamp}.mp4")
output_bbox_path = os.path.join(output_dir, f"B_bbox_ultra_{time_stamp}.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_blur = cv2.VideoWriter(output_blur_path, fourcc, fps, (origin_w, origin_h))
out_bbox = cv2.VideoWriter(output_bbox_path, fourcc, fps, (origin_w, origin_h))

cv2.namedWindow('Ultra High-End Privacy System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Ultra High-End Privacy System', 1280, 720)

absolute_prev_mask = None
mask_hold_counter = 0  
frame_times = [] 
max_confs = []  
evaluation_results = [] # 프레임별 초정밀 채점 데이터 저장 저장소

class_names = {0: "Person", 24: "Backpack", 26: "Umbrella", 28: "Handbag"}

print(f"👑 울트라 최고 사양 기준 투트랙 정밀 채점 모드 실험 가동 하위 폴더: {output_dir}")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time() 

    h, w, _ = frame.shape

    # [최고 사양화] 입력 이미지 해상도를 1280으로 확정하여 원거리 및 미세 객체 탐지율 극대화
    results = model(frame, stream=True, classes=[0, 24, 26, 28], conf=0.25, imgsz=1280, verbose=False)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    has_mask = False
    
    current_frame_bboxes = []
    max_conf_in_frame = 0.0  

    # 정답 도화지 레이어 초기화 (1. BBox 기준, 2. 순수 사람 마스크 기준)
    gt_bbox_mask = np.zeros((h, w), dtype=np.uint8)
    gt_pure_person_mask = np.zeros((h, w), dtype=np.uint8)

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                b_coords = box.xyxy[0].cpu().numpy().astype(int)  
                b_conf = float(box.conf[0].cpu().numpy())  
                b_cls = int(box.cls[0].cpu().numpy())  
                
                if b_conf > max_conf_in_frame:
                    max_conf_in_frame = b_conf

                current_frame_bboxes.append({
                    "coords": b_coords,
                    "conf": b_conf,
                    "class_id": b_cls
                })

                # 메인 지표용: 원본 해상도 스케일의 BBox 영역 기록
                x1, y1, x2, y2 = b_coords
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    gt_bbox_mask[y1:y2, x1:x2] = 255

        if r.masks is not None:
            has_mask = True
            for mask in r.masks.data:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0.4).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
                
                # 보조 지표용: 원본 해상도 스케일의 순수 사람 세그멘테이션 마스크 기록
                gt_pure_person_mask = cv2.bitwise_or(gt_pure_person_mask, binary_mask)

    max_confs.append(max_conf_in_frame)

    # 연속성 방어 홀드 카운터 시스템 상향 조정
    if has_mask:
        absolute_prev_mask = combined_mask.copy()
        mask_hold_counter = 7  
    elif not has_mask and mask_hold_counter > 0:
        if absolute_prev_mask is not None:
            combined_mask = absolute_prev_mask.copy()
            has_mask = True
        mask_hold_counter -= 1  

    # --- [스트림 1: M 접두사용 최고 사양 정밀 블러 처리 엔진] ---
    blur_only_frame = frame.copy()
    if has_mask:
        # [최고 사양화] 1280 이미지 스케일에 맞게 외곽선 확장 커널을 정방형 41x41로 대형화
        dilation_kernel = np.ones((41, 41), np.uint8)
        expanded_mask = cv2.dilate(combined_mask, dilation_kernel, iterations=1)

        # [최고 사양화] 가우시안 블러 크기를 101x101 대형 필터로 교체하여 식별 한계 완전 차단
        blurred_frame = cv2.GaussianBlur(blur_only_frame, (101, 101), 0)
        idx = (expanded_mask == 255)
        blur_only_frame[idx] = blurred_frame[idx]

    out_blur.write(blur_only_frame)

    # --- [스트림 2: B 접두사용 순수 원본 베이스 BBox 렌더링 엔진] ---
    pure_bbox_frame = frame.copy()
    y_offset = 50  
    for info in current_frame_bboxes:
        x1, y1, x2, y2 = info["coords"]
        c_score = info["conf"]
        c_id = info["class_id"]
        label_text = f"Class {c_id}({class_names.get(c_id, 'Unknown')}): Conf {c_score:.4f}"
        
        cv2.rectangle(pure_bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(pure_bbox_frame, label_text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30 

    out_bbox.write(pure_bbox_frame)

    # --- [실시간 프레임별 최고사양 기준 채점 연산] ---
    if has_mask:
        dilation_kernel = np.ones((41, 41), np.uint8)
        actual_blur_mask = cv2.dilate(combined_mask, dilation_kernel, iterations=1)
    else:
        actual_blur_mask = np.zeros((h, w), dtype=np.uint8)

    # 1) 메인 지표 계산 (BBox 기준)
    total_bbox_pixels = np.sum(gt_bbox_mask > 0)
    if total_bbox_pixels > 0:
        masked_bbox_pixels = np.sum((gt_bbox_mask > 0) & (actual_blur_mask > 0))
        bbox_masking_rate = masked_bbox_pixels / total_bbox_pixels
    else:
        masked_bbox_pixels = 0
        bbox_masking_rate = 0.0

    # 2) 보조 지표 계산 (순수 사람 형태 마스크 기준)
    total_mask_pixels = np.sum(gt_pure_person_mask > 0)
    if total_mask_pixels > 0:
        masked_pure_pixels = np.sum((gt_pure_person_mask > 0) & (actual_blur_mask > 0))
        mask_masking_rate = masked_pure_pixels / total_mask_pixels
    else:
        masked_pure_pixels = 0
        mask_masking_rate = 0.0

    end_time = time.time() 
    proc_time = end_time - start_time
    frame_times.append(proc_time) 

    # 최고사양 정량 데이터 리스트 수집
    evaluation_results.append({
        "frame": frame_idx,
        "total_bbox_pixels": int(total_bbox_pixels),
        "masked_bbox_pixels": int(masked_bbox_pixels),
        "bbox_masking_rate": float(bbox_masking_rate),
        "total_mask_pixels": int(total_mask_pixels),
        "masked_pure_pixels": int(masked_pure_pixels),
        "mask_masking_rate": float(mask_masking_rate),
        "process_time_sec": float(proc_time),
        "max_confidence": float(max_conf_in_frame),
        "success": bbox_masking_rate > 0.5  
    })

    # 실시간 모니터링 디스플레이 출력 처리
    debug_window_frame = blur_only_frame.copy()
    y_offset_win = 50
    for info in current_frame_bboxes:
        x1, y1, x2, y2 = info["coords"]
        c_score = info["conf"]
        c_id = info["class_id"]
        label_text = f"Class {c_id}({class_names.get(c_id, 'Unknown')}): Conf {c_score:.4f}"
        cv2.rectangle(debug_window_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_window_frame, label_text, (30, y_offset_win), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset_win += 30

    frame_output = cv2.resize(debug_window_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Ultra High-End Privacy System', frame_output)

    frame_idx += 1
    if cv2.waitKey(33) & 0xFF == 27: 
        break

cap.release()
out_blur.release()
out_bbox.release()
cv2.destroyAllWindows()
print(f"최고사양 비식별화 절차 마감 및 최종 결과 분석 보고서 생성 시작")

# --- [최종 결과 요약 행 병합 및 파일 출력] ---
if len(evaluation_results) > 0:
    result_df = pd.DataFrame(evaluation_results)
    
    valid_bbox_df = result_df[result_df["total_bbox_pixels"] > 0]
    valid_mask_df = result_df[result_df["total_mask_pixels"] > 0]
    
    if len(valid_bbox_df) > 0:
        tot_bbox = valid_bbox_df["total_bbox_pixels"].sum()
        tot_mas_bbox = valid_bbox_df["masked_bbox_pixels"].sum()
        overall_bbox_rate = tot_mas_bbox / tot_bbox if tot_bbox > 0 else 0
        succ_rate = valid_bbox_df["success"].mean()
    else:
        overall_bbox_rate, succ_rate = 0.0, 0.0
        
    if len(valid_mask_df) > 0:
        tot_mask = valid_mask_df["total_mask_pixels"].sum()
        tot_mas_pure = valid_mask_df["masked_pure_pixels"].sum()
        overall_mask_rate = tot_mas_pure / tot_mask if tot_mask > 0 else 0
    else:
        overall_mask_rate = 0.0
        
    total_elapsed_time = sum(frame_times)
    avg_frame_time = total_elapsed_time / len(frame_times) if len(frame_times) > 0 else 0
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    valid_confs = [c for c in max_confs if c > 0.0]
    avg_confidence = sum(valid_confs) / len(valid_confs) if len(valid_confs) > 0 else 0.0
    
    # 보고서 요약 행 병합 데이터프레임 구조화
    summary_rows = [
        {"frame": "---", "total_bbox_pixels": "---", "masked_bbox_pixels": "---", "bbox_masking_rate": "---", "total_mask_pixels": "---", "masked_pure_pixels": "---", "mask_masking_rate": "---", "process_time_sec": "---", "max_confidence": "---", "success": "---"},
        {"frame": "정답 BBox 내부 평균 마스킹 처리율(%)", "total_bbox_pixels": "", "masked_bbox_pixels": "", "bbox_masking_rate": float(overall_bbox_rate * 100), "total_mask_pixels": "", "masked_pure_pixels": "", "mask_masking_rate": "", "process_time_sec": "", "max_confidence": "", "success": ""},
        {"frame": "보조 지표: 사람 형태(Mask) 평균 처리율(%)", "total_bbox_pixels": "", "masked_bbox_pixels": "", "bbox_masking_rate": "", "total_mask_pixels": "", "masked_pure_pixels": "", "mask_masking_rate": float(overall_mask_rate * 100), "process_time_sec": "", "max_confidence": "", "success": ""},
        {"frame": "프레임 기준 데이터 확보율(%)", "total_bbox_pixels": "", "masked_bbox_pixels": "", "bbox_masking_rate": float(succ_rate * 100), "total_mask_pixels": "", "masked_pure_pixels": "", "mask_masking_rate": "", "process_time_sec": "", "max_confidence": "", "success": ""},
        {"frame": "시스템 전체 총 소요 시간(초)", "total_bbox_pixels": "", "masked_bbox_pixels": "", "bbox_masking_rate": float(total_elapsed_time), "total_mask_pixels": "", "masked_pure_pixels": "", "mask_masking_rate": "", "process_time_sec": "", "max_confidence": "", "success": ""},
        {"frame": "평균 영상 처리 속도(FPS)", "total_bbox_pixels": "", "masked_bbox_pixels": "", "bbox_masking_rate": float(avg_fps), "total_mask_pixels": "", "masked_pure_pixels": "", "mask_masking_rate": "", "process_time_sec": "", "max_confidence": "", "success": ""},
        {"frame": "탐지된 객체 평균 신뢰도 점수", "total_bbox_pixels": "", "masked_bbox_pixels": "", "bbox_masking_rate": float(avg_confidence), "total_mask_pixels": "", "masked_pure_pixels": "", "mask_masking_rate": "", "process_time_sec": "", "max_confidence": "", "success": ""}
    ]
    summary_df = pd.DataFrame(summary_rows)
    final_excel_df = pd.concat([result_df, summary_df], ignore_index=True)
    
    csv_name = os.path.join(output_dir, f"result_ultra_{time_stamp}.csv")
    json_name = os.path.join(output_dir, f"summary_ultra_{time_stamp}.json")
    
    final_excel_df.to_csv(csv_name, index=False, encoding="utf-8-sig")
    
    summary_json = {
        "bbox_internal_masking_rate_percent": float(overall_bbox_rate * 100),
        "pure_mask_masking_rate_percent": float(overall_mask_rate * 100),
        "frame_success_rate_percent": float(succ_rate * 100),
        "total_process_time_sec": float(total_elapsed_time),
        "average_fps": float(avg_fps),
        "average_bbox_confidence": float(avg_confidence)  
    }
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)
        
    print("\n========== 👑 [울트라 최고 사양] 실험 결과 보고 ========== ")
    print(f"정답 BBox 내부 평균 마스킹 처리율: {overall_bbox_rate * 100:.2f}%")
    print(f"보조 지표: 사람 형태(Mask) 평균 처리율: {overall_mask_rate * 100:.2f}%")
    print(f"프레임 기준 데이터 확보율: {succ_rate * 100:.2f}%")
    print(f"시스템 전체 총 소요 시간: {total_elapsed_time:.2f}초")
    print(f"평균 영상 처리 속도: {avg_fps:.2f} FPS")
    print(f"탐지된 객체 평균 신뢰도(Confidence): {avg_confidence:.4f}")
    print(f"저장된 최상위 결과 폴더 위치: {output_dir}")
    print("========================================================\n")