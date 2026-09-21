# pii-web: build the SPA, serve it from nginx.
#
# The node stage exists only at build time. The runtime image is nginx plus
# static files -- no node, no npm, nothing that resolves a package at runtime,
# which is what makes it shippable into an air-gapped site as a docker load
# tarball.

FROM node:20-slim AS builder

WORKDIR /build
COPY package.json package-lock.json* ./
# `npm ci` when a lockfile is present, `npm install` otherwise. A site
# building from a source tarball may not have one.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY tsconfig.json vite.config.ts tailwind.config.js postcss.config.js index.html ./
COPY src ./src
RUN npm run build

# ---------------------------------------------------------------------------

FROM nginx:1.27-alpine AS runtime

COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8081

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8081/ || exit 1
