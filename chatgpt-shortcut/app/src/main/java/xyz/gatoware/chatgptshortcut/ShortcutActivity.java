package xyz.gatoware.chatgptshortcut;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;

public class ShortcutActivity extends Activity implements RecognitionListener {
    private SpeechRecognizer recognizer;
    private TextView transcript;
    private boolean handledResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER);
        int pad = dp(28);
        root.setPadding(pad, pad, pad, pad);

        TextView listening = new TextView(this);
        listening.setText("Listening…");
        listening.setTextSize(30f);
        listening.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        listening.setGravity(Gravity.CENTER);
        root.addView(listening, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        transcript = new TextView(this);
        transcript.setText("Say what you want ChatGPT to do.");
        transcript.setTextSize(18f);
        transcript.setGravity(Gravity.CENTER);
        transcript.setPadding(0, dp(18), 0, 0);
        root.addView(transcript, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        setContentView(root);

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Open ChatGPT Shortcut once and grant microphone permission first.", Toast.LENGTH_LONG).show();
            startActivity(new Intent(this, MainActivity.class));
            finish();
            return;
        }

        new Handler(Looper.getMainLooper()).postDelayed(this::startRecognition, 180);
    }

    private void startRecognition() {
        try {
            if (android.os.Build.VERSION.SDK_INT >= 31 && SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) {
                recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(this);
            } else {
                recognizer = SpeechRecognizer.createSpeechRecognizer(this);
            }
            recognizer.setRecognitionListener(this);

            Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
            intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3);
            intent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, false);
            recognizer.startListening(intent);
        } catch (Exception e) {
            Toast.makeText(this, "Speech recognition unavailable: " + e.getClass().getSimpleName(), Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void handleText(String text) {
        if (handledResult || text == null || text.trim().isEmpty()) return;
        handledResult = true;
        String prompt = text.trim();
        transcript.setText(prompt);
        TaskState.savePending(this, prompt);

        Intent launch = getPackageManager().getLaunchIntentForPackage("com.openai.chatgpt");
        if (launch == null) {
            TaskState.clearPending(this);
            Toast.makeText(this, "ChatGPT is not installed.", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED);
        startActivity(launch);
        finish();
    }

    @Override public void onReadyForSpeech(Bundle params) {}
    @Override public void onBeginningOfSpeech() {}
    @Override public void onRmsChanged(float rmsdB) {}
    @Override public void onBufferReceived(byte[] buffer) {}
    @Override public void onEndOfSpeech() {}

    @Override
    public void onError(int error) {
        if (!handledResult) {
            Toast.makeText(this, "Didn't catch that. Try again.", Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    @Override
    public void onResults(Bundle results) {
        ArrayList<String> values = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        if (values != null && !values.isEmpty()) handleText(values.get(0));
        else onError(SpeechRecognizer.ERROR_NO_MATCH);
    }

    @Override
    public void onPartialResults(Bundle partialResults) {
        ArrayList<String> values = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        if (values != null && !values.isEmpty()) transcript.setText(values.get(0));
    }

    @Override public void onEvent(int eventType, Bundle params) {}

    @Override
    protected void onDestroy() {
        if (recognizer != null) {
            recognizer.cancel();
            recognizer.destroy();
            recognizer = null;
        }
        super.onDestroy();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
