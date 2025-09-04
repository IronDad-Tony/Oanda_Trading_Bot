import "./globals.css";
import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Live Trading Dashboard",
  description: "Next.js UI for live trading system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-dvh bg-gray-50 text-gray-900 dark:bg-gray-950 dark:text-gray-100">
        <div className="flex min-h-dvh">
          <aside className="hidden md:flex w-60 flex-col border-r border-gray-200 dark:border-gray-800 p-4 gap-2">
            <div className="text-lg font-semibold mb-2">Trading Control</div>
            <nav className="flex flex-col gap-1">
              <Link className="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-900" href="/">Dashboard</Link>
              <Link className="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-900" href="/symbols">Symbols & Model</Link>
              <Link className="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-900" href="/risk">Risk Controls</Link>
              <Link className="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-900" href="/account">Account</Link>
              <Link className="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-900" href="/history">History</Link>
              <Link className="px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-900" href="/logs">Logs</Link>
            </nav>
            <div className="mt-auto text-xs text-gray-500">API: {process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}</div>
          </aside>
          <main className="flex-1 p-4 md:p-6 w-full">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}

