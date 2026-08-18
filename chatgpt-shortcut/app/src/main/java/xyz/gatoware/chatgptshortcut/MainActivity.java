package xyz.gatoware.chatgptshortcut;

import android.Manifest;
import android.app.Activity;
import android.app.NotificationManager;
import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.text.TextUtils;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int pad = dp(20);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("ChatGPT Shortcut");
        title.setTextSize(28f);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        root.addView(title, matchWrap());

        TextView intro = new TextView(this);
        intro.setText("Hold both volume keys → dictate → the installed ChatGPT app opens, receives the prompt, sends it, then returns Home. No OpenAI API key is used.\n\nOne-time setup:");
        intro.setTextSize(16f);
        intro.setPadding(0, dp(12), 0, dp(8));
        root.addView(intro, matchWrap());

        root.addView(button("1. Grant microphone + notification permission", v -> requestNeededPermissions()), matchWrap());
        root.addView(button("2. Enable ChatGPT automation service", v -> openAccessibilitySettings()), matchWrap());
        root.addView(button("3. Enable completion notification access", v -> openNotificationAccess()), matchWrap());
        root.addView(button("4. Choose “Hold volume keys” shortcut", v -> {
            Toast.makeText(this, "In Accessibility, choose ChatGPT Shortcut and set its shortcut to Hold volume keys.", Toast.LENGTH_LONG).show();
            openAccessibilitySettings();
        }), matchWrap());
        root.addView(button("Test dictation now", v -> startActivity(new Intent(this, ShortcutActivity.class))), matchWrap());

        status = new TextView(this);
        status.setTextSize(14f);
        status.setPadding(0, dp(18), 0, 0);
        root.addView(status, matchWrap());

        TextView note = new TextView(this);
        note.setText("Note: automation intentionally has UI access only to com.openai.chatgpt. Completion detection mirrors a notification posted by ChatGPT after the prompt is sent; if ChatGPT does not post one for a particular response, its own normal notification behavior is the fallback.");
        note.setTextSize(12f);
        note.setPadding(0, dp(18), 0, 0);
        root.addView(note, matchWrap());

        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshStatus();
    }

    private void requestNeededPermissions() {
        List<String> missing = new ArrayList<>();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.RECORD_AUDIO);
        }
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (missing.isEmpty()) {
            Toast.makeText(this, "Permissions already granted.", Toast.LENGTH_SHORT).show();
            return;
        }
        requestPermissions(missing.toArray(new String[0]), 100);
    }

    private void openAccessibilitySettings() {
        try {
            startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
        } catch (Exception e) {
            Toast.makeText(this, "Could not open Accessibility settings.", Toast.LENGTH_SHORT).show();
        }
    }

    private void openNotificationAccess() {
        try {
            startActivity(new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS));
        } catch (Exception e) {
            startActivity(new Intent(Settings.ACTION_SETTINGS));
        }
    }

    private void refreshStatus() {
        boolean mic = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
        boolean notifications = Build.VERSION.SDK_INT < 33 || checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED;
        boolean accessibility = isAccessibilityServiceEnabled();
        NotificationManager nm = getSystemService(NotificationManager.class);
        boolean listener = nm != null && nm.isNotificationListenerAccessGranted(new ComponentName(this, ChatGptNotificationListener.class));
        boolean chatgpt = getPackageManager().getLaunchIntentForPackage("com.openai.chatgpt") != null;

        status.setText("Status\n" +
                "ChatGPT installed: " + yesNo(chatgpt) + "\n" +
                "Microphone: " + yesNo(mic) + "\n" +
                "App notifications: " + yesNo(notifications) + "\n" +
                "Automation service: " + yesNo(accessibility) + "\n" +
                "Notification access: " + yesNo(listener));
    }

    private boolean isAccessibilityServiceEnabled() {
        String enabled = Settings.Secure.getString(getContentResolver(), Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES);
        if (TextUtils.isEmpty(enabled)) return false;
        String target = new ComponentName(this, ChatGptAutomationService.class).flattenToString();
        TextUtils.SimpleStringSplitter splitter = new TextUtils.SimpleStringSplitter(':');
        splitter.setString(enabled);
        for (String s : splitter) {
            if (target.equalsIgnoreCase(s)) return true;
        }
        return false;
    }

    private Button button(String text, android.view.View.OnClickListener listener) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setOnClickListener(listener);
        return b;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private String yesNo(boolean v) {
        return v ? "✓" : "✗";
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
