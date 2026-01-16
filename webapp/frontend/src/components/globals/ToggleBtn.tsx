type ToggleBtnProps<T extends string | boolean> = {
    cat1: T;
    cat2: T;
    toToggle: T;
    setToToggle: (cat: T) => void;
}

export default function ToggleBtn<T extends string | boolean>({ cat1, cat2, toToggle, setToToggle }: ToggleBtnProps<T>) {

    const handleToggleClick = () => {
        setToToggle(toToggle === cat1 ? cat2 : cat1);
    }

    return (
        <div className={`relative w-14 h-7 rounded-full border border-c3 cursor-pointer ${toToggle === cat2 && 'bg-c5'}`}
            onClick={handleToggleClick}
        >
            <div className={`absolute top-0 translate-y-0.75 w-5 h-5 rounded-full bg-c1 transition-transform duration-300 ease-in-out ${toToggle === cat1 ? 'translate-x-1' : 'translate-x-7.5'}`}>
            </div>
        </div>
    )
}