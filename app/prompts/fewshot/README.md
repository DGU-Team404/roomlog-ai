# Few-shot 예시 이미지

`fewshot.json` 목록에 있는 예시 이미지가 GPT Vision 탐지 호출에 자동 포함된다.
`fewshot.json`이 없으면 few-shot 없이 zero-shot으로 동작한다.

## fewshot.json 형식

```json
[
  {
    "file": "defect_ex_1.jpg",
    "defect": "SCRATCH",
    "severity": "LOW",
    "description": "목재 표면의 대각선 방향 긁힘. ..."
  },
  {
    "file": "not_defect_1.jpg",
    "defect": "NOT_A_DEFECT",
    "severity": "LOW",
    "description": "조명에 의한 그림자. 경계가 부드럽고 표면 손상이 없음"
  }
]
```

- `file`: 이 폴더 안의 이미지 파일명 (.jpg .jpeg .png)
- `defect`: 하자 유형 (SCRATCH, CRACK, PEELING, STAIN, BREAKAGE). hard negative는 `NOT_A_DEFECT`
- `severity`: 심각도 (LOW, MEDIUM, HIGH). 자가 수리 가능 판정에 쓰이므로 심각도 눈금 보정 역할
- `description`: 판별 근거 설명
- 배열 순서대로 프롬프트에 들어가며, `[example] {defect} ({severity}): {description}` 형태로 조합되어 전송된다
- 이미지는 `detail: high`로 전송되므로 하자가 화면 대부분을 차지하는 크롭 권장

## 주의

- JSON 문법 오류나 목록에 없는 파일 참조 시 서버 시작 시점에 에러가 난다 (조용히 zero-shot으로 빠지지 않음). 수정 후 로컬에서 임포트 확인 권장
- 로드는 서버 시작 시 1회이므로, 배포 서버에서 내용을 바꾸면 재시작 필요
- 하자 예시와 함께 그림자, 벽지 무늬 등 하자로 착각하기 쉬운 hard negative를 섞어야 false positive 억제 효과가 있다
