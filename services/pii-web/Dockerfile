# pii-web: build the SPA, serve it from nginx.
#
# The node stage exists only at build time. The runtime image is nginx plus
# static files -- no node, no npm, nothing that resolves a package at runtime,
# which is what makes it shippable into an air-gapped site as a docker load
# tarball.

FROM node:20-slim AS builder

# Building behind an internal mirror and/or a TLS-intercepting proxy, which is
# the normal case at the sites this ships to -- the same seam the Python images
# have for PyPI:
#
#   docker build \
#     --secret id=npm_ca,src=/etc/ssl/certs/corporate-ca.pem \
#     --build-arg NPM_CONFIG_REGISTRY=https://nexus.internal/repository/npm/ \
#     .
#
# Both are optional. With no secret and no build arg the build uses the public
# registry and the image's own trust store, so nothing here is required in a
# normal network.
ARG NPM_CONFIG_REGISTRY=""
ENV NPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY}

WORKDIR /build
COPY package.json package-lock.json* ./

# `npm ci` when a lockfile is present, `npm install` otherwise. A site building
# from a source tarball may not have one.
#
# The `test` is not belt-and-braces. npm 10 can fail every single fetch --
# "SELF_SIGNED_CERT_IN_CHAIN" against an intercepting proxy is the way it
# happens here -- print "Exit handler never called!", and still exit 0. Without
# this check that builds a runtime image whose only symptom is "tsc: not found"
# three layers later, or worse, an image that builds and serves a stale bundle.
# An install that did not install is a failed build, so say so here.
RUN --mount=type=secret,id=npm_ca \
    if [ -s /run/secrets/npm_ca ]; then \
        export NODE_EXTRA_CA_CERTS=/run/secrets/npm_ca NPM_CONFIG_CAFILE=/run/secrets/npm_ca; \
    fi; \
    if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; \
    else npm install --no-audit --no-fund; fi; \
    test -x node_modules/.bin/tsc && test -x node_modules/.bin/vite \
      || { echo "npm install did not produce the build toolchain; see the log above" >&2; exit 1; }

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
