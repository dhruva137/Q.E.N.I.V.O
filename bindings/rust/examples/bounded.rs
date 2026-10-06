fn main() {
    match qenivo::bounded_lp() {
        Ok(obj) => println!("optimum {obj}"),
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
    match qenivo::two_binary() {
        Ok(obj) => println!("milp {obj}"),
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
