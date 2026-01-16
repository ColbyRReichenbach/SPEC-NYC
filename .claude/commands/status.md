Check project status and find the next task to work on.

1. Read current state: `cat .claude/state.yaml`
2. Find next incomplete task: `grep -n "^\s*- \[ \]" docs/NYC_IMPLEMENTATION_PLAN.md | head -5`
3. Show which phase we're in and what needs to be done next
4. Suggest which agent to use: @data-engineer, @ml-engineer, @ai-security, or @validator
