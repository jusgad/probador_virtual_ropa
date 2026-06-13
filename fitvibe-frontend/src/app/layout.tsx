import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FitVibe | Vestidor Virtual IA Premium",
  description: "Experimenta el futuro de la moda con estimación de pose por IA y recomendaciones personalizadas de tallas en tiempo real.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="es"
      className={`${inter.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex flex-col bg-[#0B0F19] text-gray-100 selection:bg-violet-500/30 selection:text-violet-200">
        {children}
      </body>
    </html>
  );
}
