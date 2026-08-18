# ChatGPT Shortcut prototype

A sideload-only Android prototype that uses the installed ChatGPT app rather than the OpenAI API.

## Intended flow

1. Hold both volume keys (configured as this app's Android accessibility shortcut).
2. Dictate a request.
3. The app launches ChatGPT.
4. A narrowly scoped Accessibility service fills the ChatGPT composer and clicks Send.
5. The app returns Home.
6. If ChatGPT posts a completion/update notification, the notification listener mirrors it as **ChatGPT finished**.

## First-run setup

Open **ChatGPT Shortcut** once and follow the numbered buttons:

1. Grant microphone and notification permission.
2. Enable **ChatGPT prompt automation** under Accessibility.
3. Grant notification access to **ChatGPT Shortcut**.
4. In Accessibility, configure **ChatGPT Shortcut** to use the **Hold volume keys** shortcut.

The Accessibility service is restricted to package `com.openai.chatgpt`.

## Prototype caveat

The ChatGPT Android UI is not a public automation API. The sender uses accessibility labels first and a geometry fallback second, so a future ChatGPT UI update can require a small tweak. Completion notification depends on ChatGPT posting a notification after the request is sent.
