NAME diesel_week_presolved
OBJSENSE
    MAX
ROWS
 N OBJ
 L unit_capacity
 E diesel_yield
 L sulphur_spec
COLUMNS
 crude_light OBJ -60.0
 crude_light unit_capacity 1.0
 crude_light diesel_yield 0.55
 crude_light sulphur_spec -0.2
 crude_heavy OBJ -45.0
 crude_heavy unit_capacity 1.0
 crude_heavy diesel_yield 0.4
 crude_heavy sulphur_spec 0.3
 diesel OBJ 130.0
 diesel diesel_yield -1.0
RHS
 RHS unit_capacity 600.0
 RHS sulphur_spec 30.0
BOUNDS
 UP BND crude_light 400.0
 UP BND crude_heavy 300.0
 LO BND diesel 250.0
ENDATA
