#!/bin/bash
# Cloud Build를 사용한 배포 (Docker 없이)

set -e

PROJECT_ID="vision-final-478501"
REGION="us-central1"
SERVICE_NAME="distilled-vision-agent"

echo "🚀 Cloud Build를 사용한 배포 시작"
echo "프로젝트: $PROJECT_ID"
echo "서비스: $SERVICE_NAME"
echo

# 1. 프로젝트 설정
echo "📋 프로젝트 설정..."
gcloud config set project $PROJECT_ID

# 2. 필요한 API 활성화
echo "🔧 API 활성화..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 3. Cloud Build로 빌드 및 배포
echo "☁️ Cloud Build로 빌드 및 배포..."
gcloud builds submit --config cloudbuild.yaml

echo
echo "✅ 배포 완료!"
echo "🌐 서비스 URL 확인 중..."

# 서비스 URL 가져오기
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)' 2>/dev/null || echo "아직 배포 중...")

if [ "$SERVICE_URL" != "아직 배포 중..." ]; then
    echo "🎮 게임 URL: $SERVICE_URL"
    echo
    echo "브라우저에서 접속하여 게임을 플레이하세요!"
    
    # macOS에서 자동으로 브라우저 열기
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🌍 브라우저에서 열기..."
        open "$SERVICE_URL"
    fi
else
    echo "⏳ 배포가 진행 중입니다. 잠시 후 다음 명령어로 URL을 확인하세요:"
    echo "gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'"
fi
