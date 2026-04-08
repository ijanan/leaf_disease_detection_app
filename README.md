# leaf_disease_detection_app

A Flutter app for leaf disease detection.

## Download APK from GitHub

This repo includes a GitHub Actions workflow that builds a **release APK** and makes it downloadable:

- **Actions artifact**: Every run uploads `app-release.apk` as an artifact in the workflow run page.
- **GitHub Release asset**: When you push a tag like `v1.0.0`, the workflow creates a GitHub Release and attaches the APK.

### Create a Release APK on GitHub

1. Create a version tag:
	- `git tag v1.0.0`
	- `git push origin v1.0.0`
2. Wait for the workflow **Build Android APK** to finish.
3. Download the APK from the **Releases** page (or from the workflow run artifacts).

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.
![alt text](image.png)