# PyPI Publishing Instructions

## Current Status
- ✅ Package built: `optimiz_rs-1.0.0-cp38-abi3-macosx_10_12_x86_64.whl`
- ✅ Package name: `optimiz-rs` (to avoid conflict with existing "optimizr")
- ❌ Need PyPI API token (username/password auth deprecated)

## Steps to Publish

### 1. Get PyPI API Token

1. Login to PyPI: https://pypi.org/account/login/
   - Username: `ThotDjehuty`
   - Password: `G2p._468pfSH73G`

2. Create API token: https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Token name: `optimiz-rs-publishing`
   - Scope: "Entire account" (can limit to project later)
   - **IMPORTANT:** Copy the token immediately (starts with `pypi-`)
   - Store securely (won't be shown again)

### 2. Upload to PyPI

```bash
cd /Users/melvinalvarez/Documents/Workspace/optimiz-r

# Upload with API token
twine upload target/wheels/optimiz_rs-1.0.0-cp38-abi3-macosx_10_12_x86_64.whl \
  -u __token__ \
  -p pypi-YOUR_TOKEN_HERE
```

**Note:** Username must be `__token__` (literal string) when using API tokens.

### 3. Verify Publication

After successful upload, verify at:
- Package page: https://pypi.org/project/optimiz-rs/
- Test install: `pip install optimiz-rs`

### 4. Update Documentation

Once published, update these files:
- `RELEASE_NOTES_v1.0.0.md` - Change "(publishing in progress)" to actual link
- `LINKEDIN_POST.md` - Update PyPI link
- Commit and push changes

## Alternative: Store Token in ~/.pypirc

For future uploads, store token securely:

```bash
# Create ~/.pypirc
cat > ~/.pypirc << 'EOF'
[pypi]
  username = __token__
  password = pypi-YOUR_TOKEN_HERE
EOF

# Secure the file
chmod 600 ~/.pypirc

# Then upload without credentials in command
twine upload target/wheels/optimiz_rs-1.0.0-*.whl
```

## Troubleshooting

**403 Forbidden:**
- PyPI deprecated username/password auth
- Must use API tokens
- Ensure username is `__token__` (not your actual username)

**Package name conflict:**
- Already handled - using `optimiz-rs`
- Cannot use `optimizr`, `optimiz-r`, or `optimizR` (all normalize to "optimizr")

**Token not working:**
- Verify token copied completely (very long string)
- Check token hasn't expired
- Ensure no extra spaces/newlines

---

**Current Publication Status:**
- ✅ crates.io: Published at https://crates.io/crates/optimiz-rs
- ⏳ PyPI: Ready to publish (just need API token)
