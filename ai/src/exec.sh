#!/bin/bash

# ============================================
# 곡명 리스트 (여기에 처리할 곡명을 입력하세요)
# ============================================
SONG_NAMES=(
    "€URO TRA$H - What You Looking For (Extended" # 상
    "Jauz & Ephwurd - Rock The Party" # 상
    "Jauz, NONSENS - The Beat (Original Mix)" # 상
    "Diplo, Skrillex, 2 Chainz - Febreze feat. 2 Chainz (Original Mix)" # bpm 높음
    "Avicii - Levels (Skrillex Remix)" # 멜로디였던걸로 기억함
    "Malaa - Notorious (Original Mix)" # low basshouse
    "Malaa - Bling Bling (Matroda Remix) (Origina"
    "Habstrakt, Badjokes - Right Here (Original Mix)"
    "Ghastly - Get Focused (Original Mix)"
    "Knock2 - REWiND (Wolfgang Gartner Remix [Extended Mix])"
)

# 설정
INDEX_DIR="./index"
MUSIC_DIR="../../Music/basshouse"
GOAL="up"
TOP_K=10

# 음악 파일 확장자
AUDIO_EXTS=("mp3" "wav" "flac" "ogg" "m4a" "aiff" "aif")

# 결과 디렉토리 생성
OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"

# 파일명에서 안전한 이름 생성 (특수문자 제거/변환)
sanitize_filename() {
    local filename="$1"
    # 확장자 제거
    filename="${filename%.*}"
    # 특수문자를 언더스코어로 변환 (파일명에 사용 가능한 문자만 남김)
    echo "$filename" | sed 's/[^a-zA-Z0-9가-힣._-]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
}

# 곡명으로 파일 찾기
find_song_file() {
    local song_name="$1"
    local dir="$2"
    
    for ext in "${AUDIO_EXTS[@]}"; do
        # 파일명에 곡명이 포함된 파일 찾기
        found=$(find "$dir" -type f -iname "*${song_name}*.${ext}" 2>/dev/null | head -n 1)
        if [ -n "$found" ]; then
            echo "$found"
            return 0
        fi
    done
    return 1
}

# 메인 처리
file_count=${#SONG_NAMES[@]}
echo "총 $file_count 개의 곡을 처리합니다."
echo ""

counter=0
for song_name in "${SONG_NAMES[@]}"; do
    counter=$((counter + 1))
    
    echo "[$counter/$file_count] 검색 중: $song_name"
    
    # 곡명으로 파일 찾기
    music_file=$(find_song_file "$song_name" "$MUSIC_DIR")
    
    if [ -z "$music_file" ]; then
        echo "  ✗ 파일을 찾을 수 없습니다: $song_name"
        echo ""
        continue
    fi
    
    # 파일명에서 곡 제목 추출
    filename=$(basename "$music_file")
    safe_name=$(sanitize_filename "$filename")
    output_file="${OUTPUT_DIR}/res_${safe_name}.txt"
    
    echo "  -> 파일: $filename"
    echo "  -> 출력: $output_file"
    
    # recommend 명령 실행 및 결과 저장
    python main.py recommend \
        --index_dir "$INDEX_DIR" \
        --current "$music_file" \
        --goal "$GOAL" \
        --top_k "$TOP_K" \
        > "./results/$output_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 완료"
    else
        echo "  ✗ 오류 발생 (결과 파일 확인: $output_file)"
    fi
    echo ""
done

echo "모든 처리가 완료되었습니다. 결과는 $OUTPUT_DIR 디렉토리에 저장되었습니다."