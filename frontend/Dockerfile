# Dashboard image: static build served by nginx; the API URL is baked in at build time.
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API=http://localhost:8000
ENV VITE_API=$VITE_API
RUN npx vite build
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
