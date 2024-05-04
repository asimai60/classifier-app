package com.example.classifierclient;

// SplashActivity.java
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;

public class SplashActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        Button classifyButton = findViewById(R.id.buttonClassify);
        classifyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Start the main activity when the "Classify" button is clicked
                startActivity(new Intent(SplashActivity.this, MainActivity.class));
                // Finish the splash activity
                finish();
            }
        });
    }
}
