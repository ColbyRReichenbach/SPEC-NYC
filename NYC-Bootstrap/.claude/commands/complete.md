Mark a task as complete in the implementation plan.

For the task described in $ARGUMENTS:

1. Find the matching `- [ ]` line in `docs/NYC_IMPLEMENTATION_PLAN.md`
2. Change it to `- [x]`
3. If this completes a phase, update `.claude/state.yaml`
4. Commit: `git commit -am "Complete: $ARGUMENTS"`
5. Show what's next
