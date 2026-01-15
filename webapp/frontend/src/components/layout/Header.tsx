import { Link, useLocation } from "react-router-dom";

import Btn1 from "../globals/btns/Btn1";
import Btn2 from "../globals/btns/Btn2";

export default function Header() {
    const location = useLocation();

    const isActive = (path: string) => {
        return location.pathname === path;
    }

    return (
        <>
            <div className="font-sans fixed top-0 w-full h-16 px-32 py-4">
                <div className="w-full h-full grid grid-cols-3">
                    
                    {/* Logo */}
                    <div className="w-full h-full col-span-1 col-start-1">
                        <Link to="/" 
                            className="w-full h-full flex flex-row items-center justify-start gap-2"
                        >
                            <div className="cursor-pointer">
                                *
                            </div>
                            <div className="font-medium text-c1 text-2xl cursor-pointer">
                                fading's forward
                            </div>
                        </Link>
                    </div>

                    {/* Pages */}
                    <div className="w-full h-full col-span-1 col-start-2">
                        <div className="w-full h-full flex flex-row items-center justify-center gap-4">
                            {["About", "EliseLM", "Pricing", "Updates", "Blog"].map((page) => (
                                <>
                                    <Link to={"/" + page}
                                        key={page}
                                        className={`hover:text-c1 cursor-pointer ${
                                            isActive('/' + page) 
                                                ? 'text-c1'
                                                : 'text-c2'
                                        }`}
                                    >
                                        {page}
                                    </Link>
                                </>
                            ))}
                        </div>
                    </div>

                    {/* Cliquables */}
                    <div className="w-full h-full col-span-1 col-start-3">
                        <div className="w-full h-full flex flex-row items-center justify-end gap-2">
                            <Btn2 text={"Contact"} />
                            <Btn1 text={"Get Started"} />
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}