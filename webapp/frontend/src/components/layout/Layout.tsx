import type { ReactNode } from "react";
import Header from "./Header";
import Footer from "./Footer";

type LayoutProps = {
    children: ReactNode;
}

export default function Layout( {children}: LayoutProps ) {
    return (
        <>
            <div className="min-h-screen w-screen bg-c4 flex flex-col items-start justify-start scrollbar-hide">
                <Header />
                <main className="flex-1 w-full pt-20 px-32">
                    {children}
                </main>
                <Footer />
            </div>
        </>
    )
}