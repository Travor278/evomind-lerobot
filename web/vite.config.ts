import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig } from 'vite';

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === 'seatbelt';

export default defineConfig(() => {
  const server = {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8767',
        ws: true,
      },
    },
    ...(isCodexSeatbeltSandbox
      ? { watch: { useFsEvents: false, usePolling: true } }
      : {}),
  };

  return {
    css: { postcss: { plugins: [tailwindcss()] } },
    server,
    plugins: [vinext()],
  };
});
