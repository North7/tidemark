// Cloudflare Pages Function：只在產品案例中文版（/case-study/）執行，其他檔案照常直接給。
// 把訪客 IP 所在的國家（Cloudflare 的 request.cf.country，例如 TW、US）寫進 <html data-geo>，
// 頁面開頭那段用它決定預設語言：網址的 ?lang= → 自己選過的 → IP 地區（TW／HK／MO／CN 用中文，其他用英文）→ 瀏覽器語言。
// 做法和 Aron Lab（North7/north7.github.io 的 functions/index.js）相同。部署時 wrangler 從倉庫根目錄的 functions/ 讀取。
export async function onRequest({ request, next }) {
  const res = await next();
  const country = request.cf && request.cf.country;
  if (!country || !/^[A-Z]{2}$/.test(country) || !(res.headers.get('content-type') || '').includes('text/html')) return res;
  return new HTMLRewriter().on('html', { element(el) { el.setAttribute('data-geo', country); } }).transform(res);
}
