# dots.tts.edit Playground

The dots.tts.edit Playground is a local React application served by the Python
backend in this directory. Build its `frontend/dist` bundle from the committed
TypeScript source before launching the application.

## Build the frontend

Node.js 20 or newer and npm are required.

```bash
cd apps/edit_playground/frontend
npm ci
npm test
npm run build
```

From the repository root, launch the local application with:

```bash
python apps/edit_playground/app.py \
  --model-name-or-path dots-studio/dots.tts.edit \
  --optimize
```

If `frontend/dist/index.html` is missing, the application exits with the build
commands instead of loading the model. To install dependencies and build the
frontend as part of an explicit launch, use:

```bash
python apps/edit_playground/app.py \
  --model-name-or-path dots-studio/dots.tts.edit \
  --optimize \
  --rebuild-frontend
```

Normal launches never invoke npm. Re-run `npm run build` after changing files
under `frontend/src`.
