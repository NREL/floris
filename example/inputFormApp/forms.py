from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, IntegerField, SelectField
from wtforms.validators import DataRequired

class InputForm(FlaskForm):
    farm_description = StringField('Description/Identifier', default='Example Wind Farm', validators=[DataRequired()])
    wind_speed = DecimalField('Wind Speed', places=1, default=8.0, validators=[DataRequired()])
    wind_direction = DecimalField('Wind Speed', places=1, default=270.0, validators=[DataRequired()])
    turbulence_intensity = DecimalField('Turbulence Intensity', places=1, default=0.1, validators=[DataRequired()])
    wind_shear = DecimalField('Wind Shear', places=2, default=0.12, validators=[DataRequired()])
    wind_veer = DecimalField('Wind Veer', places=1, default=0.0, validators=[DataRequired()])   
    air_density = DecimalField('Air Density', places=3, default=1.225, validators=[DataRequired()]) 
    wake_combination = SelectField('Wake Combination', default='sosfs', choices=[('sosfs', 'sosfs')], validators=[DataRequired()])

    turbine_description = StringField('Description/Identifier', default='NREL 5MW', validators=[DataRequired()])
    rotor_diameter = DecimalField('Rotor Diameter', places=1, default=126.0, validators=[DataRequired()])
    hub_height = DecimalField('Hub Height', places=1, default=90, validators=[DataRequired()])
    blade_count = IntegerField('Blade Count', default=3, validators=[DataRequired()])
    pP = DecimalField('pP', places=2, default=1.88, validators=[DataRequired()])
    pT = DecimalField('pT', places=2, default=1.88, validators=[DataRequired()])
    generator_efficiency = DecimalField('Generator Efficiency', places=1, default=1.0, validators=[DataRequired()]) 
    eta = DecimalField('eta', places=3, default=0.768, validators=[DataRequired()]) 
    blade_pitch = DecimalField('Blade Pitch', places=1, default=1.9, validators=[DataRequired()])
    yaw_angle = DecimalField('Yaw Angle', places=1, default=20.0, validators=[DataRequired()])
    tilt_angle = DecimalField('Tilt Angle', places=1, default=0.0, validators=[DataRequired()]) 
    TSR = DecimalField('TSR', places=2, default=8.0, validators=[DataRequired()])
     
    velocity_model = SelectField('Velocity Model', choices=[('Jensen', 'Jensen'), ('FLORIS', 'FLORIS'), ('gauss_velocity', 'gauss_velocity')], validators=[DataRequired()])
    deflection_model = SelectField('Velocity Model', choices=[('Jimenez', 'Jimenez'), ('gauss_deflection', 'gauss_deflection')], validators=[DataRequired()])
     
    jensen_we = DecimalField('we', places=2, default=0.05, validators=[DataRequired()])
    floris_me_1 = DecimalField('me Zone 1', places=2, default=-0.05, validators=[DataRequired()])
    floris_me_2 = DecimalField('me Zone 2', places=2, default=0.3, validators=[DataRequired()])
    floris_me_3 = DecimalField('me Zone 3', places=2, default=1.0, validators=[DataRequired()])
    floris_aU = DecimalField('aU', places=1, default=12.0, validators=[DataRequired()])
    floris_bU = DecimalField('bU', places=1, default=1.3, validators=[DataRequired()])
    floris_mU_1 = DecimalField('mU Zone 1', places=1, default=0.5, validators=[DataRequired()])
    floris_mU_2 = DecimalField('mU Zone 2', places=1, default=1.0, validators=[DataRequired()])
    floris_mU_3 = DecimalField('mU Zone 3', places=1, default=5.5, validators=[DataRequired()])
    gauss_v_placeholder = DecimalField('Placeholder', places=1, default=0, validators=[DataRequired()])
    guass_d_placeholder = DecimalField('Placeholder', places=1, default=0, validators=[DataRequired()])
    jimenez_kd = DecimalField('kd', places=2, default=0.17, validators=[DataRequired()])
    jimenez_ad = DecimalField('ad', places=2, default=-4.5, validators=[DataRequired()])
    jimenez_bd = DecimalField('bd', places=2, default=-0.01, validators=[DataRequired()])
