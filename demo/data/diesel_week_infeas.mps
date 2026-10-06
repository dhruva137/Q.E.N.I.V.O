NAME diesel_infeas
OBJSENSE
    MAX
ROWS
 N OBJ
 L unit_capacity
 E diesel_yield
 G diesel_contract
 L sulphur_spec
COLUMNS
 crude_light OBJ -60 unit_capacity 1
 crude_light diesel_yield 0.55 sulphur_spec -0.2
 crude_heavy OBJ -45 unit_capacity 1
 crude_heavy diesel_yield 0.40 sulphur_spec 0.3
 diesel OBJ 130 diesel_yield -1
 diesel diesel_contract 1
RHS
 RHS unit_capacity 600 diesel_contract 500
 RHS sulphur_spec 30
BOUNDS
 UP BND crude_light 400
 UP BND crude_heavy 300
ENDATA
