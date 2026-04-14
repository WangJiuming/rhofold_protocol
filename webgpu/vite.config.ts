import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  server: {
    headers: {
      // Required for SharedArrayBuffer (ORT-Web WASM multi-threading)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      // Allow serving files from parent directory (for example data)
      allow: ['..'],
    },
  },
  plugins: [
    {
      name: 'serve-models-and-data',
      configureServer(server) {
        // Serve ONNX models from webgpu/models/
        server.middlewares.use('/models', (req, res, next) => {
          const filePath = path.resolve(__dirname, 'models', req.url!.slice(1) || '');
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          import('fs').then(fs => {
            if (fs.existsSync(filePath)) {
              const stream = fs.createReadStream(filePath);
              res.setHeader('Content-Type', 'application/octet-stream');
              stream.pipe(res);
            } else {
              next();
            }
          });
        });
        // Serve example data
        server.middlewares.use('/example', (req, res, next) => {
          const filePath = path.resolve(__dirname, '..', 'data', 'rhofold', '3owz_A', req.url!.slice(1) || '');
          import('fs').then(fs => {
            if (fs.existsSync(filePath)) {
              res.setHeader('Content-Type', 'text/plain');
              fs.createReadStream(filePath).pipe(res);
            } else {
              next();
            }
          });
        });
      },
    },
  ],
  optimizeDeps: {
    // onnxruntime-web ships WASM binaries that Vite's optimizer can't process
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
  assetsInclude: ['**/*.onnx'],
});
