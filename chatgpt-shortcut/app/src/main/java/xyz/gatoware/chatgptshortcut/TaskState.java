package xyz.gatoware.chatgptshortcut;

import android.content.Context;
import android.content.SharedPreferences;

final class TaskState {
    private static final String PREFS = "task_state";
    private static final String PENDING_PROMPT = "pending_prompt";
    private static final String PENDING_AT = "pending_at";
    private static final String AWAITING = "awaiting_completion";
    private static final String SENT_AT = "sent_at";

    private TaskState() {}

    static void savePending(Context context, String prompt) {
        prefs(context).edit()
                .putString(PENDING_PROMPT, prompt)
                .putLong(PENDING_AT, System.currentTimeMillis())
                .putBoolean(AWAITING, false)
                .apply();
    }

    static String pendingPrompt(Context context) {
        SharedPreferences p = prefs(context);
        long age = System.currentTimeMillis() - p.getLong(PENDING_AT, 0L);
        if (age < 0 || age > 120_000L) {
            clearPending(context);
            return null;
        }
        return p.getString(PENDING_PROMPT, null);
    }

    static void clearPending(Context context) {
        prefs(context).edit().remove(PENDING_PROMPT).remove(PENDING_AT).apply();
    }

    static void markSent(Context context) {
        prefs(context).edit()
                .remove(PENDING_PROMPT)
                .remove(PENDING_AT)
                .putBoolean(AWAITING, true)
                .putLong(SENT_AT, System.currentTimeMillis())
                .apply();
    }

    static boolean awaitingCompletion(Context context) {
        SharedPreferences p = prefs(context);
        if (!p.getBoolean(AWAITING, false)) return false;
        long age = System.currentTimeMillis() - p.getLong(SENT_AT, 0L);
        if (age < 0 || age > 6 * 60 * 60 * 1000L) {
            clearAwaiting(context);
            return false;
        }
        return true;
    }

    static long sentAt(Context context) {
        return prefs(context).getLong(SENT_AT, 0L);
    }

    static void clearAwaiting(Context context) {
        prefs(context).edit().putBoolean(AWAITING, false).remove(SENT_AT).apply();
    }

    private static SharedPreferences prefs(Context context) {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }
}
