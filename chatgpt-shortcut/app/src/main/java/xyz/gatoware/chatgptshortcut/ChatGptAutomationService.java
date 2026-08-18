package xyz.gatoware.chatgptshortcut;

import android.accessibilityservice.AccessibilityService;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ChatGptAutomationService extends AccessibilityService {
    static final String CHANNEL_ID = "chatgpt_shortcut";
    private final Handler handler = new Handler(Looper.getMainLooper());
    private boolean automationRunning;
    private int retries;

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        ensureChannel();
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (event == null || event.getPackageName() == null) return;
        if (!"com.openai.chatgpt".contentEquals(event.getPackageName())) return;
        if (TaskState.pendingPrompt(this) == null || automationRunning) return;

        automationRunning = true;
        retries = 0;
        handler.postDelayed(this::attemptSend, 900);
    }

    private void attemptSend() {
        String prompt = TaskState.pendingPrompt(this);
        if (prompt == null) {
            automationRunning = false;
            return;
        }

        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) {
            retryOrFail();
            return;
        }

        AccessibilityNodeInfo editor = findEditor(root);
        if (editor == null) {
            retryOrFail();
            return;
        }

        boolean filled = setText(editor, prompt);
        if (!filled) {
            retryOrFail();
            return;
        }

        handler.postDelayed(() -> {
            AccessibilityNodeInfo freshRoot = getRootInActiveWindow();
            AccessibilityNodeInfo freshEditor = freshRoot == null ? null : findEditor(freshRoot);
            AccessibilityNodeInfo send = freshRoot == null ? null : findSendButton(freshRoot, freshEditor);
            boolean sent = send != null && send.performAction(AccessibilityNodeInfo.ACTION_CLICK);

            if (sent) {
                retries = 0;
                TaskState.markSent(this);
                notifyLocal("Sent to ChatGPT", "Your dictated task was submitted.");
                handler.postDelayed(() -> performGlobalAction(GLOBAL_ACTION_HOME), 650);
                automationRunning = false;
            } else {
                retries = 0;
                notifyLocal("Prompt filled", "I couldn't confidently find ChatGPT's Send button. Tap Send manually this time.");
                TaskState.clearPending(this);
                automationRunning = false;
            }
        }, 550);
    }

    private void retryOrFail() {
        if (retries++ < 10) {
            handler.postDelayed(this::attemptSend, 450);
            return;
        }
        retries = 0;
        automationRunning = false;
        notifyLocal("ChatGPT automation paused", "I couldn't find ChatGPT's message box. Open a normal chat and try the shortcut again.");
    }

    private AccessibilityNodeInfo findEditor(AccessibilityNodeInfo root) {
        AccessibilityNodeInfo focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
        if (isUsableEditor(focused)) return focused;

        List<AccessibilityNodeInfo> editors = new ArrayList<>();
        collectEditors(root, editors);
        if (editors.isEmpty()) return null;

        AccessibilityNodeInfo best = null;
        int bestBottom = Integer.MIN_VALUE;
        for (AccessibilityNodeInfo node : editors) {
            Rect r = new Rect();
            node.getBoundsInScreen(r);
            if (node.isVisibleToUser() && r.bottom > bestBottom) {
                bestBottom = r.bottom;
                best = node;
            }
        }
        return best != null ? best : editors.get(editors.size() - 1);
    }

    private void collectEditors(AccessibilityNodeInfo node, List<AccessibilityNodeInfo> out) {
        if (node == null) return;
        if (isUsableEditor(node)) out.add(node);
        for (int i = 0; i < node.getChildCount(); i++) {
            collectEditors(node.getChild(i), out);
        }
    }

    private boolean isUsableEditor(AccessibilityNodeInfo node) {
        if (node == null || !node.isVisibleToUser()) return false;
        if (node.isEditable()) return true;
        for (AccessibilityNodeInfo.AccessibilityAction action : node.getActionList()) {
            if (action.getId() == AccessibilityNodeInfo.ACTION_SET_TEXT) return true;
        }
        return false;
    }

    private boolean setText(AccessibilityNodeInfo editor, String prompt) {
        editor.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
        Bundle args = new Bundle();
        args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, prompt);
        if (editor.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) return true;

        ClipboardManager clipboard = getSystemService(ClipboardManager.class);
        if (clipboard == null) return false;
        clipboard.setPrimaryClip(ClipData.newPlainText("ChatGPT prompt", prompt));
        editor.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
        return editor.performAction(AccessibilityNodeInfo.ACTION_PASTE);
    }

    private AccessibilityNodeInfo findSendButton(AccessibilityNodeInfo root, AccessibilityNodeInfo editor) {
        List<AccessibilityNodeInfo> clickable = new ArrayList<>();
        collectClickable(root, clickable);

        for (AccessibilityNodeInfo node : clickable) {
            String label = (safe(node.getText()) + " " + safe(node.getContentDescription()) + " " + safe(node.getViewIdResourceName())).toLowerCase(Locale.ROOT);
            if ((label.contains("send") || label.contains("submit")) && !label.contains("feedback")) {
                return node;
            }
        }

        if (editor == null) return null;
        Rect er = new Rect();
        editor.getBoundsInScreen(er);
        AccessibilityNodeInfo best = null;
        long bestScore = Long.MAX_VALUE;

        for (AccessibilityNodeInfo node : clickable) {
            if (node == editor || !node.isVisibleToUser()) continue;
            Rect r = new Rect();
            node.getBoundsInScreen(r);
            if (r.width() <= 0 || r.height() <= 0) continue;

            boolean verticalOverlap = r.bottom >= er.top && r.top <= er.bottom;
            boolean toRight = r.centerX() >= er.centerX();
            if (!verticalOverlap || !toRight) continue;

            long dx = Math.abs((long) r.centerX() - er.right);
            long dy = Math.abs((long) r.centerY() - er.centerY());
            long area = (long) r.width() * r.height();
            long score = dx * 5 + dy * 3 + area / 50;
            if (score < bestScore) {
                bestScore = score;
                best = node;
            }
        }
        return best;
    }

    private void collectClickable(AccessibilityNodeInfo node, List<AccessibilityNodeInfo> out) {
        if (node == null) return;
        if (node.isClickable() && node.isVisibleToUser()) out.add(node);
        for (int i = 0; i < node.getChildCount(); i++) {
            collectClickable(node.getChild(i), out);
        }
    }

    private String safe(CharSequence text) {
        return text == null ? "" : text.toString();
    }

    private String safe(String text) {
        return text == null ? "" : text;
    }

    private void ensureChannel() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null) {
            nm.createNotificationChannel(new NotificationChannel(CHANNEL_ID, "ChatGPT Shortcut", NotificationManager.IMPORTANCE_DEFAULT));
        }
    }

    private void notifyLocal(String title, String text) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm == null) return;
        ensureChannel();
        android.app.Notification notification = new android.app.Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle(title)
                .setContentText(text)
                .setAutoCancel(true)
                .build();
        nm.notify((int) (System.currentTimeMillis() & 0x7fffffff), notification);
    }

    @Override
    public void onInterrupt() {}
}
