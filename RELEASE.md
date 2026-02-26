# GitHub Release (Android APK)

This repo is set up to build a **signed release APK** in **GitHub Actions** and attach it to a **GitHub Release** automatically.

## One-time setup

### 1) Confirm your keystore file

You need your Android keystore file (usually `*.jks` or `*.keystore`).

### 2) Add GitHub Secrets

In your GitHub repo:

**Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

- `ANDROID_KEYSTORE_BASE64`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

#### Create `ANDROID_KEYSTORE_BASE64` on Windows (PowerShell)

Run this command (replace the path to your keystore):

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\\path\\to\\your-release.jks")) | Set-Content -Encoding ascii keystore_base64.txt
```

Open `keystore_base64.txt`, copy all text, paste into the GitHub secret `ANDROID_KEYSTORE_BASE64`.

## Create a release

### Option A (recommended): push a version tag

1) Choose a version tag like `v1.0.0`
2) Run:

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will:
- build `app-release.apk`
- create a GitHub Release
- upload the APK to that release

### Option B: manual run

Go to **Actions → Android Release APK → Run workflow**.

## Download the APK

GitHub repo → **Releases** → open the version → download the attached `app-release.apk`.

## Troubleshooting

- If the workflow fails at "Missing secret …", double-check the secrets are added.
- If signing fails, verify your keystore passwords/alias are correct.
- If you want a real public app, change the Android `applicationId` to your own unique id.
