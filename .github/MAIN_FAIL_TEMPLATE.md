---
title: "⚠ {{ env.BUILD_TYPE }} failed on `main`"
labels: "CI failure"
---

Commit {{ sha }} by @{{ payload.sender.login }} did not pass CI.
{% if env.DEPENDENCY_CHANGES %}

> [!warning]
> A dependency version changed between the baseline and the contender. This
> can be the cause of a benchmark regression.

```text
{{ env.DEPENDENCY_CHANGES }}
```

{% endif %}

> [!note]
> This issue was created automatically as a notification.
> Don't edit its title or close it until the failure is resolved.
> Otherwise, a new identical issue will be opened.
>
> Instead, consider opening new issue(s) with a more descriptive title and link
> them to this issue.
