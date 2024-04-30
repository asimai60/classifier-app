package com.example.classifierclient;

import android.os.Bundle;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import java.io.IOException;
import android.os.Environment;
import java.io.File;
import java.io.FileOutputStream;
import androidx.activity.EdgeToEdge;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.widget.TextView;
import org.json.JSONException;
import org.json.JSONObject;

import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;


import java.io.FileNotFoundException;

import android.graphics.Color;
import android.os.Handler;
import android.util.TypedValue;


public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST_CODE = 101;
    private static final int IMAGE_CAPTURE_CODE = 102;
    private Uri imageUri;
    private TextView resultTextView;
    private static final String DEFAULT_TEXT = "Take a picture for classification";
    private static final int DELAY_MILLISECONDS = 5000; // 5 seconds delay


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultTextView = findViewById(R.id.resultTextView);
        Button captureButton = findViewById(R.id.button_capture);
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(getApplicationContext(),
                        Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{
                            Manifest.permission.CAMERA,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE
                    }, CAMERA_REQUEST_CODE);
                } else {
                    openCamera();
                }
            }
        });
    }

    private void openCamera() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION, "From the Camera");
        imageUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(cameraIntent, IMAGE_CAPTURE_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            Toast.makeText(this, "Camera permission is required to use camera.", Toast.LENGTH_SHORT).show();
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == IMAGE_CAPTURE_CODE && resultCode == RESULT_OK) {
            try {
                // Get the captured image bitmap
                Bitmap imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);

                // Get the dimensions of the ImageView
                ImageView imageView = findViewById(R.id.imageView);
                int targetWidth = imageView.getWidth();
                int targetHeight = imageView.getHeight();

                // Display the preview image in the ImageView
                displayPreviewImage(imageBitmap);

                // Save a scaled-down version of the image for preview
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(imageBitmap, targetWidth, targetHeight, true);
                File previewFile = saveBitmapToFile(scaledBitmap); // Save the scaled-down image to a file

                // Send the full-size image to the server
                sendImage(imageUri);
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private File saveBitmapToFile(Bitmap bitmap) {
        // Create a file to save the image
        File file = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "preview.jpg");
        try (FileOutputStream out = new FileOutputStream(file)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return file;
    }

    private void displayPreviewImage(Bitmap bitmap) {
        ImageView imageView = findViewById(R.id.imageView);
        imageView.setImageBitmap(bitmap);
    }
    private void sendImage(Uri imageUri) {
        Bitmap imageBitmap = null;
        try {
            imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            Toast.makeText(this, "File not found", Toast.LENGTH_SHORT).show();
            return;
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            return;
        }

        // Create a file from the URI and send it to the server
        File file = convertBitmapToFile(imageBitmap);

        OkHttpClient client = new OkHttpClient();
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", file.getName(),
                        RequestBody.create(MediaType.parse("image/jpeg"), file))
                .build();
        Request request = new Request.Builder()
                .url("http://10.100.102.9:5000/upload")
                .post(requestBody)
                .build();
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                e.printStackTrace(); // This will print detailed error message in the log
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to send image: " + e.getMessage(), Toast.LENGTH_LONG).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (!response.isSuccessful()) {
                    throw new IOException("Unexpected code " + response);
                } else {
                    // Convert the response body to string
                    String responseData = response.body().string();

                    // Use the main thread to update the UI
                    runOnUiThread(() -> {
                        try {
                            // Convert the response string to JSON
                            JSONObject jsonObject = new JSONObject(responseData);
                            String label = jsonObject.getString("label");

                            // Display the classification result in a TextView
                            displayResult("Classification Result: " + label);
//                            TextView resultView = findViewById(R.id.resultTextView);
//                            resultView.setText("Classification Result: " + label);
                        } catch (JSONException e) {
                            e.printStackTrace();
                            Toast.makeText(MainActivity.this, "Failed to parse response", Toast.LENGTH_SHORT).show();
                        }
                    });
                }
            }
        });
    }

    private File convertBitmapToFile(Bitmap bitmap) {
        // Create a file to save the image
        File file = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "upload.jpg");
        try (FileOutputStream out = new FileOutputStream(file)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return file;
    }

    public String getPathFromUri(Uri uri) {
        String path = null;
        Cursor cursor = getContentResolver().query(uri, null, null, null, null);
        if (cursor != null) {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            path = cursor.getString(idx);
            cursor.close();
        }
        return path;
    }

    private void displayResult(String result) {
        // Display the result in green, big letters
        //if the result's last word is unknown, display it in red
        resultTextView.setText(result);
        String[] words = result.split(" ");
        if (words[words.length - 1].equals("unknown")) {
            resultTextView.setTextColor(Color.RED);
        }
        else {
            resultTextView.setTextColor(Color.GREEN);
        }
        
        resultTextView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24);

        // Reset to default text after 5 seconds
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                resultTextView.setText(DEFAULT_TEXT);
                resultTextView.setTextColor(Color.BLACK); // Set default text color
                resultTextView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18); // Set default text size
            }
        }, DELAY_MILLISECONDS);
    }
}