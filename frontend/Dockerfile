# RAGify frontend image - serves static HTML via Nginx
FROM nginx:1.27-alpine

WORKDIR /usr/share/nginx/html

# Copy static HTML pages and assets
COPY chat.html index.html knowledge.html settings.html ./
COPY src ./src
COPY public/vite.svg ./vite.svg

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
