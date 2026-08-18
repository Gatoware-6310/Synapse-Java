package xyz.gatoware.chatgptshortcut;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.os.Bundle;
import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;

public class ChatGptNotificationListener extends NotificationListenerService {
    @Override
    public void onNotificationPosted(StatusBarNotification sbn) {
        if (sbn == null || !"com.openai.chatgpt".equals(sbn.getPackageName())) return;
        if (!TaskState.awaitingCompletion(this)) return;
        if (sbn.getPostTime() < TaskState.sentAt(this) + 1_000L) return;

        Bundle extras = sbn.getNotification().extras;
        CharSequence title = extras.getCharSequence(Notification.EXTRA_TITLE);
        CharSequence text = extras.getCharSequence(Notification.EXTRA_TEXT);
        String body = text != null && text.length() > 0
                ? text.toString()
                : (title != null && title.length() > 0 ? title.toString() : "Your ChatGPT task has an update.");

        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm == null) return;
        nm.createNotificationChannel(new NotificationChannel(ChatGptAutomationService.CHANNEL_ID, "ChatGPT Shortcut", NotificationManager.IMPORTANCE_DEFAULT));

        Intent launch = getPackageManager().getLaunchIntentForPackage("com.openai.chatgpt");
        PendingIntent pending = null;
        if (launch != null) {
            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED);
            pending = PendingIntent.getActivity(this, 0, launch, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        }

        Notification.Builder builder = new Notification.Builder(this, ChatGptAutomationService.CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("ChatGPT finished")
                .setContentText(body)
                .setStyle(new Notification.BigTextStyle().bigText(body))
                .setAutoCancel(true);
        if (pending != null) builder.setContentIntent(pending);

        nm.notify(6310, builder.build());
        TaskState.clearAwaiting(this);
    }
}
