import pandas as pd
import glob
import os

# -------------------------------------------------------------
# [설정 영역] 결과 CSV 파일들이 모여있는 폴더 경로를 적어주세요
folder_path = r"C:\Users\leeju\OneDrive\바탕 화면\SW보안 프로젝트\코드\실험"
# -------------------------------------------------------------

output_folder_name = "experiment_summary_results"
target_output_dir = os.path.join(folder_path, output_folder_name)

if not os.path.exists(target_output_dir):
    os.makedirs(target_output_dir)

file_pattern = os.path.join(folder_path, "*.csv")
all_files = glob.glob(file_pattern)

# 기존 생성된 요약 리포트 파일들은 대상에서 제외
all_files = [f for f in all_files if "Comparison" not in f and "Summary" not in f and "Report" not in f]

if len(all_files) == 0:
    print("설정 오류: 지정한 폴더 내에 처리할 CSV 파일이 존재하지 않습니다.")
    exit()

print(f"데이터 분석 진행 중: 총 {len(all_files)}개 파일 조사 중...")

summary_records = []

for file_path in all_files:
    file_name = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
        df_clean = df[pd.to_numeric(df['frame'], errors='coerce').notnull()].copy()
        
        if len(df_clean) == 0:
            continue
            
        metrics = ["process_time_sec", "mask_masking_rate", "bbox_masking_rate", "max_confidence", "total_mask_pixels"]
        for col in metrics:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 사람이 존재하는 유효 프레임만 필터링
        if "total_mask_pixels" in df_clean.columns:
            df_clean = df_clean[df_clean["total_mask_pixels"] > 0].copy()
        else:
            df_clean = df_clean[df_clean["mask_masking_rate"] > 0].copy()
            
        if len(df_clean) == 0:
            continue
                
        avg_time = df_clean["process_time_sec"].mean()
        file_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        summary_records.append({
            "영상 파일명": file_name,
            "평균 속도(FPS)": file_fps,
            "최저 속도(Seconds)": df_clean["process_time_sec"].min(),
            "최고 속도(Seconds)": df_clean["process_time_sec"].max(),
            "평균 사람 형태 차단율(%)": df_clean["mask_masking_rate"].mean() * 100,
            "최저 사람 형태 차단율(%)": df_clean["mask_masking_rate"].min() * 100,
            "최고 사람 형태 차단율(%)": df_clean["mask_masking_rate"].max() * 100,
            "평균 BBox 마스킹률(%)": df_clean["bbox_masking_rate"].mean() * 100,
            "최저 BBox 마스킹률(%)": df_clean["bbox_masking_rate"].min() * 100,
            "최고 BBox 마스킹률(%)": df_clean["bbox_masking_rate"].max() * 100,
            "평균 신뢰도 점수(Conf)": df_clean["max_confidence"].mean()
        })
    except Exception as e:
        print(f"파일 읽기 패스 ({file_name}): {e}")

df_summary = pd.DataFrame(summary_records)

if len(df_summary) == 0:
    print("데이터 오류: 사람이 감지된 유효한 데이터가 존재하지 않습니다.")
    exit()

# 각 영상별 '평균값' 항목들 사이에서 최저 평균과 최고 평균을 유발한 영상 추적
f_min_fps = df_summary.loc[df_summary["평균 속도(FPS)"].idxmin(), "영상 파일명"]
f_max_fps = df_summary.loc[df_summary["평균 속도(FPS)"].idxmax(), "영상 파일명"]

f_min_sec_avg = df_summary.loc[df_summary["최저 속도(Seconds)"].idxmin(), "영상 파일명"]
f_max_sec_avg = df_summary.loc[df_summary["최저 속도(Seconds)"].idxmax(), "영상 파일명"]
f_abs_min_sec = df_summary.loc[df_summary["최고 속도(Seconds)"].idxmin(), "영상 파일명"]
f_abs_max_sec = df_summary.loc[df_summary["최고 속도(Seconds)"].idxmax(), "영상 파일명"]

f_min_mask_avg = df_summary.loc[df_summary["평균 사람 형태 차단율(%)"].idxmin(), "영상 파일명"]
f_max_mask_avg = df_summary.loc[df_summary["평균 사람 형태 차단율(%)"].idxmax(), "영상 파일명"]
f_min_mask_low = df_summary.loc[df_summary["최저 사람 형태 차단율(%)"].idxmin(), "영상 파일명"]
f_max_mask_low = df_summary.loc[df_summary["최저 사람 형태 차단율(%)"].idxmax(), "영상 파일명"]
f_min_mask_high = df_summary.loc[df_summary["최고 사람 형태 차단율(%)"].idxmin(), "영상 파일명"]
f_max_mask_high = df_summary.loc[df_summary["최고 사람 형태 차단율(%)"].idxmax(), "영상 파일명"]

f_min_bbox_avg = df_summary.loc[df_summary["평균 BBox 마스킹률(%)"].idxmin(), "영상 파일명"]
f_max_bbox_avg = df_summary.loc[df_summary["평균 BBox 마스킹률(%)"].idxmax(), "영상 파일명"]
f_min_bbox_low = df_summary.loc[df_summary["최저 BBox 마스킹률(%)"].idxmin(), "영상 파일명"]
f_max_bbox_low = df_summary.loc[df_summary["최저 BBox 마스킹률(%)"].idxmax(), "영상 파일명"]
f_min_bbox_high = df_summary.loc[df_summary["최고 BBox 마스킹률(%)"].idxmin(), "영상 파일명"]
f_max_bbox_high = df_summary.loc[df_summary["최고 BBox 마스킹률(%)"].idxmax(), "영상 파일명"]

f_min_conf = df_summary.loc[df_summary["평균 신뢰도 점수(Conf)"].idxmin(), "영상 파일명"]
f_max_conf = df_summary.loc[df_summary["평균 신뢰도 점수(Conf)"].idxmax(), "영상 파일명"]

header_row = {col: col for col in df_summary.columns}
blank_row = {col: "" for col in df_summary.columns}

# 요약 통계 행 생성 (평균값 기준 스탯 명시)
grand_total_rows = [
    {
        "영상 파일명": "[전체 종합 평균값] Total_Batch_Average",
        "평균 속도(FPS)": f"{df_summary['평균 속도(FPS)'].mean():.2f}",
        "최저 속도(Seconds)": f"{df_summary['최저 속도(Seconds)'].mean():.4f}",
        "최고 속도(Seconds)": f"{df_summary['최고 속도(Seconds)'].mean():.4f}",
        "평균 사람 형태 차단율(%)": f"{df_summary['평균 사람 형태 차단율(%)'].mean():.2f}%",
        "최저 사람 형태 차단율(%)": f"{df_summary['최저 사람 형태 차단율(%)'].mean():.2f}%",
        "최고 사람 형태 차단율(%)": f"{df_summary['최고 사람 형태 차단율(%)'].mean():.2f}%",
        "평균 BBox 마스킹률(%)": f"{df_summary['평균 BBox 마스킹률(%)'].mean():.2f}%",
        "최저 BBox 마스킹률(%)": f"{df_summary['최저 BBox 마스킹률(%)'].mean():.2f}%",
        "최고 BBox 마스킹률(%)": f"{df_summary['최고 BBox 마스킹률(%)'].mean():.2f}%",
        "평균 신뢰도 점수(Conf)": f"{df_summary['평균 신뢰도 점수(Conf)'].mean():.4f}"
    },
    {
        "영상 파일명": "[최저 평균 레코드] 각 지표별 가장 낮은 평균 수치를 기록한 영상 추적",
        "평균 속도(FPS)": f"{df_summary['평균 속도(FPS)'].min():.2f} (from: {f_min_fps})",
        "최저 속도(Seconds)": f"{df_summary['최저 속도(Seconds)'].min():.4f} (from: {f_min_sec_avg})",
        "최고 속도(Seconds)": f"{df_summary['최고 속도(Seconds)'].min():.4f} (from: {f_abs_min_sec})", 
        "평균 사람 형태 차단율(%)": f"{df_summary['평균 사람 형태 차단율(%)'].min():.2f}% (from: {f_min_mask_avg})",
        "최저 사람 형태 차단율(%)": f"{df_summary['최저 사람 형태 차단율(%)'].min():.2f}% (from: {f_min_mask_low})",
        "최고 사람 형태 차단율(%)": f"{df_summary['최고 사람 형태 차단율(%)'].min():.2f}% (from: {f_min_mask_high})",
        "평균 BBox 마스킹률(%)": f"{df_summary['평균 BBox 마스킹률(%)'].min():.2f}% (from: {f_min_bbox_avg})",
        "최저 BBox 마스킹률(%)": f"{df_summary['최저 BBox 마스킹률(%)'].min():.2f}% (from: {f_min_bbox_low})",
        "최고 BBox 마스킹률(%)": f"{df_summary['최고 BBox 마스킹률(%)'].min():.2f}% (from: {f_min_bbox_high})",
        "평균 신뢰도 점수(Conf)": f"{df_summary['평균 신뢰도 점수(Conf)'].min():.4f} (from: {f_min_conf})"
    },
    {
        "영상 파일명": "[최고 평균 레코드] 각 지표별 가장 높은 평균 수치를 기록한 영상 추적",
        "평균 속도(FPS)": f"{df_summary['평균 속도(FPS)'].max():.2f} (from: {f_max_fps})",
        "최저 속도(Seconds)": f"{df_summary['최저 속도(Seconds)'].max():.4f} (from: {f_max_sec_avg})",
        "최고 속도(Seconds)": f"{df_summary['최고 속도(Seconds)'].max():.4f} (from: {f_abs_max_sec})",
        "평균 사람 형태 차단율(%)": f"{df_summary['평균 사람 형태 차단율(%)'].max():.2f}% (from: {f_max_mask_avg})",
        "최저 사람 형태 차단율(%)": f"{df_summary['최저 사람 형태 차단율(%)'].max():.2f}% (from: {f_max_mask_low})",
        "최고 사람 형태 차단율(%)": f"{df_summary['최고 사람 형태 차단율(%)'].max():.2f}% (from: {f_max_mask_high})",
        "평균 BBox 마스킹률(%)": f"{df_summary['평균 BBox 마스킹률(%)'].max():.2f}% (from: {f_max_bbox_avg})",
        "최저 BBox 마스킹률(%)": f"{df_summary['최저 BBox 마스킹률(%)'].max():.2f}% (from: {f_max_bbox_high})",
        "최고 BBox 마스킹률(%)": f"{df_summary['최고 BBox 마스킹률(%)'].max():.2f}% (from: {f_max_bbox_high})",
        "평균 신뢰도 점수(Conf)": f"{df_summary['평균 신뢰도 점수(Conf)'].max():.4f} (from: {f_max_conf})"
    }
]

df_grand_total = pd.DataFrame(grand_total_rows)

# 레이아웃 최종 병합
df_final_master = pd.concat([
    df_summary, 
    pd.DataFrame([blank_row]), 
    pd.DataFrame([header_row]), 
    df_grand_total
], ignore_index=True)

# 순수 CSV 파일로 내보내기 (한글 깨짐 방지를 위해 utf-8-sig 적용)
output_master_path = os.path.join(target_output_dir, "Total_Batch_Files_Summary_Report.csv")
df_final_master.to_csv(output_master_path, index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print("분석 완료: 통계가 각 영상의 평균 수치를 기준으로 재구축되었습니다.")
print(f"결과 파일 저장 완료 주소: {output_master_path}")
print("========================================================\n")