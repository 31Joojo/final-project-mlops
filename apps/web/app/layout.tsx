import "./globals.css";

export const metadata = {
  title: "MLOps Final Project",
  description: "Next.js UI for FastAPI inference"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
