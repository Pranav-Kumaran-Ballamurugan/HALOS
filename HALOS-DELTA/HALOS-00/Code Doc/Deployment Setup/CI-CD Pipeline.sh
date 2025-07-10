# Sample CI step
- name: Code Analysis
  run: |
    curl -X POST $HALOS_URL/checkcode \
      -H "Content-Type: application/json" \
      -d '{"code": "${{ steps.source.outputs.code }}"}'