# Paper Desk frontend

React + TypeScript + Vite + Tailwind single-page app for the chat API in `src/api/`. See `docs/chatbot.md`.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api to python -m src.api on :8000
npm run build    # dist/, served by the API at http://127.0.0.1:8000
npm run lint
```

`src/types.ts` mirrors `src/api/schemas.py`; change both together.
