function out = model
%
% ARC_field_mangiarotti.m
%
% Model exported on Jun 26 2026, 12:33 by COMSOL 6.4.0.293.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath(fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM'));

model.label('ARC_field_mangiarotti.mph');

model.param.set('wid', '12[mm]');
model.param.set('Nt', '2000');
model.param.set('thk', '0.32[mm]');
model.param.set('Ip', '350[A]');
model.param.set('Heig', '(wid+dist)*Nc');
model.param.set('Nc', '12');
model.param.set('Ntf', '18');
model.param.set('L1', '7.24[m]');
model.param.set('R1', '0.30[m]');
model.param.set('R10', '0.70[m]');
model.param.set('R0', 'R10+R1/2+L1/4+Nt*thk', [native2unicode(hex2dec({'59' '27'}), 'unicode')  native2unicode(hex2dec({'53' '4a'}), 'unicode')  native2unicode(hex2dec({'5f' '84'}), 'unicode') ]);
model.param.set('a', 'R1/2+L1/4', [native2unicode(hex2dec({'5c' '0f'}), 'unicode')  native2unicode(hex2dec({'53' '4a'}), 'unicode')  native2unicode(hex2dec({'5f' '84'}), 'unicode') ]);
model.param.set('dist', '3[mm]');
model.param.set('theta_0.5H', '2*asin(Heig/4/R10)');
model.param.set('dr_TF', 'thk*Nt');
model.param.set('amp_turns', 'Nt*Nc*Ip');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 3);

model.component('comp1').curvedInterior(false);

model.result.table.create('tbl1', 'Table');
model.result.table.create('tbl2', 'Table');

model.func.create('int1', 'Interpolation');
model.func.create('int2', 'Interpolation');
model.func.create('int3', 'Interpolation');
model.func.create('int4', 'Interpolation');
model.func.create('int5', 'Interpolation');
model.func.create('int6', 'Interpolation');
model.component('comp1').func.create('int7', 'Interpolation');
model.func('int1').active(false);
model.func('int1').set('funcname', 'IcB30');
model.func('int1').set('table', {'0' '2264';  ...
'1' '1273';  ...
'2' '974';  ...
'3' '803';  ...
'4' '686';  ...
'5' '596';  ...
'6' '523';  ...
'7' '462';  ...
'8' '410';  ...
'9' '363';  ...
'10' '324';  ...
'11' '289';  ...
'12' '258';  ...
'13' '230'});
model.func('int1').set('filecolumns', 2);
model.func('int1').set('columnKeys', {'col1' 'col2'});
model.func('int1').set('fununit', {''});
model.func('int1').set('argunit', {''});
model.func('int2').active(false);
model.func('int2').set('funcname', 'IcB20');
model.func('int2').set('table', {'0' '2540';  ...
'0.5' '1781';  ...
'1' '1376';  ...
'1.5' '1187';  ...
'2' '1037';  ...
'2.5' '945';  ...
'3' '872';  ...
'3.5' '806';  ...
'4' '751';  ...
'4.5' '711';  ...
'5' '664';  ...
'5.5' '631';  ...
'6' '609';  ...
'6.5' '577';  ...
'7' '562';  ...
'8' '549';  ...
'9' '538';  ...
'10' '529'});
model.func('int2').set('filecolumns', 2);
model.func('int2').set('columnKeys', {'col1' 'col2'});
model.func('int2').set('funcnames', {'col1' 'int2' 'col2' 'int1'});
model.func('int2').set('fununit', {''});
model.func('int2').set('argunit', {''});
model.func('int3').active(false);
model.func('int3').set('funcname', 'IcB4');
model.func('int3').set('table', {'0' '4434';  ...
'1' '3295';  ...
'2' '2601';  ...
'3' '2195';  ...
'4' '1912';  ...
'5' '1702';  ...
'6' '1540';  ...
'7' '1403';  ...
'8' '1293';  ...
'9' '1192';  ...
'10' '1100';  ...
'11' '1040';  ...
'12' '975';  ...
'13' '916'});
model.func('int3').set('filecolumns', 2);
model.func('int3').set('columnKeys', {'col1' 'col2'});
model.func('int3').set('funcnames', {'col1' 'int3' 'col2' 'int1'});
model.func('int3').set('fununit', {''});
model.func('int3').set('argunit', {''});
model.func('int4').active(false);
model.func('int4').set('funcname', 'IcB65_1');
model.func('int4').set('table', {'0' '800'; '1' '450'; '2' '140'});
model.func('int4').set('filecolumns', 2);
model.func('int4').set('columnKeys', {'col1' 'col2'});
model.func('int4').set('funcnames', {'col1' 'int4' 'col2' 'int1'});
model.func('int4').set('fununit', {''});
model.func('int4').set('argunit', {''});
model.func('int5').active(false);
model.func('int5').set('funcname', 'IcB65');
model.func('int5').set('table', {'0' '800'; '1' '240'; '2' '140'; '3' '100'});
model.func('int5').set('filecolumns', 2);
model.func('int5').set('columnKeys', {'col1' 'col2'});
model.func('int5').set('funcnames', {'col1' 'int5' 'col2' 'int1'});
model.func('int5').set('interp', 'piecewisecubic');
model.func('int5').set('extrap', 'linear');
model.func('int5').set('fununit', {''});
model.func('int5').set('argunit', {''});
model.func('int6').active(false);
model.func('int6').set('funcname', 'IcB50a');
model.func('int6').set('table', {'0' '1104';  ...
'1' '355';  ...
'2' '238';  ...
'3' '186';  ...
'4' '151';  ...
'5' '125'});
model.func('int6').set('interp', 'piecewisecubic');
model.func('int6').set('fununit', {''});
model.func('int6').set('argunit', {''});
model.component('comp1').func('int7').set('funcname', 'IcB20');
model.component('comp1').func('int7').set('table', {'0' '2540';  ...
'0.5' '1781';  ...
'1' '1376';  ...
'1.5' '1187';  ...
'2' '1037';  ...
'2.5' '945';  ...
'3' '872';  ...
'3.5' '806';  ...
'4' '751';  ...
'4.5' '711';  ...
'5' '664';  ...
'5.5' '631';  ...
'6' '609';  ...
'6.5' '577';  ...
'7' '562';  ...
'8' '549';  ...
'9' '538';  ...
'10' '529';  ...
'' ''});
model.component('comp1').func('int7').set('fununit', {'A'});

model.component('comp1').mesh.create('mesh1');

model.component('comp1').geom('geom1').lengthUnit('cm');
model.component('comp1').geom('geom1').geomRep('cadps');
model.component('comp1').geom('geom1').designBooleans(false);
model.component('comp1').geom('geom1').create('wp1', 'WorkPlane');
model.component('comp1').geom('geom1').feature('wp1').set('quickz', '-Heig/2');
model.component('comp1').geom('geom1').feature('wp1').set('unite', true);
model.component('comp1').geom('geom1').feature('wp1').geom.create('ls1', 'LineSegment');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('specify1', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('coord1', {'R10' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('specify2', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls1').set('coord2', {'R10' 'L1/2'});
model.component('comp1').geom('geom1').feature('wp1').geom.create('ls2', 'LineSegment');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('specify1', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('coord1', {'R10+Nt*thk' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('specify2', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls2').set('coord2', {'R10+Nt*thk' 'L1/2'});
model.component('comp1').geom('geom1').feature('wp1').geom.create('ca1', 'CircularArc');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca1').set('center', {'R10+Nt*thk+R1' 'L1/2'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca1').set('r', 'R1');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca1').set('angle1', 90);
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca1').set('angle2', 180);
model.component('comp1').geom('geom1').feature('wp1').geom.create('ca2', 'CircularArc');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca2').set('center', {'R10+Nt*thk+R1' 'L1/2'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca2').set('r', 'R1+Nt*thk');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca2').set('angle1', 90);
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca2').set('angle2', 180);
model.component('comp1').geom('geom1').feature('wp1').geom.create('ca3', 'CircularArc');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca3').set('center', {'R10+Nt*thk+R1' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca3').set('r', 'R1+L1/2');
model.component('comp1').geom('geom1').feature('wp1').geom.create('ca4', 'CircularArc');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca4').set('center', {'R10+Nt*thk+R1' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ca4').set('r', 'R1+L1/2+Nt*thk');
model.component('comp1').geom('geom1').feature('wp1').geom.create('ls3', 'LineSegment');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls3').set('specify1', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls3').set('coord1', {'R10' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls3').set('specify2', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls3').set('coord2', {'R10+Nt*thk' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.create('ls4', 'LineSegment');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls4').set('specify1', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls4').set('coord1', {'R10+L1/2+R1*2+Nt*thk' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls4').set('specify2', 'coord');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('ls4').set('coord2', {'R10+L1/2+R1*2+2*Nt*thk' '0'});
model.component('comp1').geom('geom1').feature('wp1').geom.create('mir1', 'Mirror');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('mir1').active(false);
model.component('comp1').geom('geom1').feature('wp1').geom.feature('mir1').set('keep', true);
model.component('comp1').geom('geom1').feature('wp1').geom.feature('mir1').set('axis', [0 1]);
model.component('comp1').geom('geom1').feature('wp1').geom.feature('mir1').selection('input').set({'ca1' 'ca2' 'ca3' 'ca4' 'ls1' 'ls2'});
model.component('comp1').geom('geom1').feature('wp1').geom.create('csol1', 'ConvertToSolid');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('csol1').selection('input').set({'ca1' 'ca2' 'ca3' 'ca4' 'ls1' 'ls2' 'ls3' 'ls4' 'mir1'});
model.component('comp1').geom('geom1').feature('wp1').geom.create('del1', 'Delete');
model.component('comp1').geom('geom1').feature('wp1').geom.feature('del1').active(false);
model.component('comp1').geom('geom1').feature('wp1').geom.feature('del1').selection('input').init(2);
model.component('comp1').geom('geom1').create('ext1', 'Extrude');
model.component('comp1').geom('geom1').feature('ext1').setIndex('distance', 'Heig/2', 0);
model.component('comp1').geom('geom1').feature('ext1').selection('input').set({'wp1'});
model.component('comp1').geom('geom1').create('wp4', 'WorkPlane');
model.component('comp1').geom('geom1').feature('wp4').label('Bcen');
model.component('comp1').geom('geom1').feature('wp4').set('unite', true);
model.component('comp1').geom('geom1').feature('wp4').geom.create('c1', 'Circle');
model.component('comp1').geom('geom1').feature('wp4').geom.feature('c1').set('r', 2);
model.component('comp1').geom('geom1').feature('wp4').geom.feature('c1').set('angle', 180);
model.component('comp1').geom('geom1').feature('wp4').geom.feature('c1').set('pos', {'R0' '0'});
model.component('comp1').geom('geom1').create('rev2', 'Revolve');
model.component('comp1').geom('geom1').feature('rev2').set('angle2', '360/Ntf/2');
model.component('comp1').geom('geom1').feature('rev2').selection('input').set({'wp4'});
model.component('comp1').geom('geom1').create('ext2', 'Extrude');
model.component('comp1').geom('geom1').feature('ext2').active(false);
model.component('comp1').geom('geom1').feature('ext2').setIndex('distance', '-Heig/2', 0);
model.component('comp1').geom('geom1').create('wp3', 'WorkPlane');
model.component('comp1').geom('geom1').feature('wp3').active(false);
model.component('comp1').geom('geom1').feature('wp3').set('quickz', 'Heig/2');
model.component('comp1').geom('geom1').feature('wp3').set('unite', true);
model.component('comp1').geom('geom1').create('wp2', 'WorkPlane');
model.component('comp1').geom('geom1').feature('wp2').set('unite', true);
model.component('comp1').geom('geom1').feature('wp2').geom.create('r1', 'Rectangle');
model.component('comp1').geom('geom1').feature('wp2').geom.feature('r1').set('size', [800 500]);
model.component('comp1').geom('geom1').feature('wp2').geom.feature('r1').set('pos', [0 0]);
model.component('comp1').geom('geom1').feature('wp2').geom.create('csol1', 'ConvertToSolid');
model.component('comp1').geom('geom1').feature('wp2').geom.feature('csol1').selection('input').set({'r1'});
model.component('comp1').geom('geom1').create('rev1', 'Revolve');
model.component('comp1').geom('geom1').feature('rev1').set('angle2', '360/Ntf/2');
model.component('comp1').geom('geom1').feature('rev1').selection('input').set({'wp2'});
model.component('comp1').geom('geom1').create('rot1', 'Rotate');
model.component('comp1').geom('geom1').feature('rot1').active(false);
model.component('comp1').geom('geom1').feature('rot1').set('axis', [0 1 0]);
model.component('comp1').geom('geom1').feature('rot1').set('rot', 'range(0,360/Ntf,360/Ntf)');
model.component('comp1').geom('geom1').feature('rot1').selection('input').set({'ext1'});
model.component('comp1').geom('geom1').create('del1', 'Delete');
model.component('comp1').geom('geom1').feature('del1').active(false);
model.component('comp1').geom('geom1').feature('del1').selection('input').init(3);
model.component('comp1').geom('geom1').feature('del1').selection('input').set('rot1(1)', 1);
model.component('comp1').geom('geom1').create('dif1', 'Difference');
model.component('comp1').geom('geom1').feature('dif1').active(false);
model.component('comp1').geom('geom1').feature('dif1').set('selresult', true);
model.component('comp1').geom('geom1').feature('dif1').selection('input').set({'rev1'});
model.component('comp1').geom('geom1').feature('dif1').selection('input2').set({'rot1'});
model.component('comp1').geom('geom1').create('rot2', 'Rotate');
model.component('comp1').geom('geom1').feature('rot2').active(false);
model.component('comp1').geom('geom1').feature('rot2').set('axis', [0 1 0]);
model.component('comp1').geom('geom1').feature('rot2').set('rot', 'range(0,360/Ntf,360)');
model.component('comp1').geom('geom1').feature('rot2').selection('input').set({'ext1'});
model.component('comp1').geom('geom1').create('cyl1', 'Cylinder');
model.component('comp1').geom('geom1').feature('cyl1').active(false);
model.component('comp1').geom('geom1').feature('cyl1').label('Air domain');
model.component('comp1').geom('geom1').feature('cyl1').set('r', 800);
model.component('comp1').geom('geom1').feature('cyl1').set('h', 1000);
model.component('comp1').geom('geom1').feature('cyl1').set('pos', [0 -500 0]);
model.component('comp1').geom('geom1').feature('cyl1').set('axis', [0 1 0]);
model.component('comp1').geom('geom1').run;
model.component('comp1').geom('geom1').run('fin');

model.component('comp1').selection.create('sel1', 'Explicit');
model.component('comp1').selection('sel1').geom('geom1', 2);
model.component('comp1').selection('sel1').label('coil surface');

model.component('comp1').variable.create('var1');
model.component('comp1').variable('var1').set('Bper', 'abs(mf.Bx*nx+mf.By*ny+mf.Bz*nz+eps)');
model.component('comp1').variable('var1').set('Ic', 'IcB20(abs(Br))');
model.component('comp1').variable('var1').set('Br', '-mf.Bx*(x<=R10+Nt*thk+eps&&abs(y)<=L1/2)+(-mf.Bx*(R10+Nt*thk+R1-x)+mf.By*(y-L1/2))/sqrt((R10+Nt*thk+R1-x)^2+(y-L1/2)^2)*(x<= R10+Nt*thk+R1&&y>L1/2)+(-mf.Bx*(R10+Nt*thk+R1-x)-mf.By*(-y-L1/2))/sqrt((R10+Nt*thk+R1-x)^2+(-y-L1/2)^2)*(x<=R10+Nt*thk+R1&&y<-L1/2)+(mf.Bx*(x-Nt*thk-R1-R10)+mf.By*y)/sqrt((x-Nt*thk-R1-R10)^2+y^2)*(x>Nt*thk+R1+R10)');
model.component('comp1').variable('var1').set('Bperp_T', 'Ip*abs(Br)/1[T]/1[A]');
model.component('comp1').variable('var1').set('Bmag_T', 'Ip*mf.normB/1[T]/1[A]');
model.component('comp1').variable('var1').set('Bpara_T', 'sqrt(max(Bmag_T^2-Bperp_T^2,0))');
model.component('comp1').variable('var1').set('thetad', 'atan2(Bpara_T,Bperp_T+1e-9)*180/pi');
model.component('comp1').variable('var1').set('fperp', '0.6781-0.0107*Bmag_T');
model.component('comp1').variable('var1').set('fpara', '0.7808-0.0105*Bmag_T');
model.component('comp1').variable('var1').set('th0m', '11.99-0.069*Bmag_T');
model.component('comp1').variable('var1').set('Jcperp', '3268.2*Bmag_T^(-0.6442)*exp(-(20-4.2)*log(1/fperp)/17.8)');
model.component('comp1').variable('var1').set('Jcpara', '(4086-71.859*Bmag_T)*exp(-(20-4.2)*log(1/fpara)/17.8)');
model.component('comp1').variable('var1').set('Jc_mang', 'Jcperp+(Jcpara-Jcperp)*exp(-(90-thetad)/th0m)');

model.component('comp1').view('view3').tag('view7');
model.component('comp1').view('view4').tag('view6');
model.component('comp1').view('view5').tag('view3');
model.view.create('view4', 3);
model.view.create('view5', 3);

model.component('comp1').material.create('mat2', 'Common');
model.component('comp1').material('mat2').propertyGroup('def').func.create('eta', 'Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func.create('Cp', 'Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func.create('rho', 'Analytic');
model.component('comp1').material('mat2').propertyGroup('def').func.create('k', 'Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func.create('cs', 'Analytic');
model.component('comp1').material('mat2').propertyGroup.create('RefractiveIndex', 'RefractiveIndex', 'Refractive index');

model.component('comp1').cpl.create('intop1', 'Integration');
model.component('comp1').cpl('intop1').selection.geom('geom1', 2);

model.component('comp1').coordSystem.create('sys2', 'Cylindrical');

model.component('comp1').physics.create('mf', 'InductionCurrents', 'geom1');
model.component('comp1').physics('mf').create('al1', 'AmperesLawFluid', 3);
model.component('comp1').physics('mf').feature('al1').selection.all;
model.component('comp1').physics('mf').feature('al1').featureInfo.create('information');
model.component('comp1').physics('mf').create('dcont1', 'Continuity', 2);
model.component('comp1').physics('mf').create('coil1', 'Coil', 3);
model.component('comp1').physics('mf').feature('coil1').selection.set([2]);
model.component('comp1').physics('mf').feature('coil1').feature('ccc1').feature('ct1').selection.set([6]);
model.component('comp1').physics('mf').feature('coil1').feature('ccc1').create('cg1', 'CoilGround', 2);
model.component('comp1').physics('mf').feature('coil1').feature('ccc1').feature('cg1').selection.set([22]);
model.component('comp1').physics('mf').feature('coil1').featureInfo.create('information');
model.component('comp1').physics('mf').create('pmc1', 'PerfectMagneticConductor', 2);
model.component('comp1').physics('mf').feature('pmc1').selection.set([1 2 8 11 15 21]);
model.component('comp1').physics('mf').create('symp1', 'SymmetryPlane', 2);
model.component('comp1').physics('mf').feature('symp1').selection.set([3 6 16 18 20 22]);

model.component('comp1').mesh('mesh1').create('ftri1', 'FreeTri');
model.component('comp1').mesh('mesh1').create('swe1', 'Sweep');
model.component('comp1').mesh('mesh1').create('conv1', 'Convert');
model.component('comp1').mesh('mesh1').create('ftet1', 'FreeTet');
model.component('comp1').mesh('mesh1').create('ftet2', 'FreeTet');
model.component('comp1').mesh('mesh1').create('ftet3', 'FreeTet');
model.component('comp1').mesh('mesh1').feature('ftri1').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature('ftri1').feature('size1').selection.geom('geom1', 2);
model.component('comp1').mesh('mesh1').feature('swe1').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('swe1').create('dis1', 'Distribution');
model.component('comp1').mesh('mesh1').feature('conv1').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('ftet1').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('ftet1').selection.set([2]);
model.component('comp1').mesh('mesh1').feature('ftet1').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature('ftet2').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('ftet2').selection.set([3]);
model.component('comp1').mesh('mesh1').feature('ftet2').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature('ftet3').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('ftet3').selection.set([1]);
model.component('comp1').mesh('mesh1').feature('ftet3').create('size1', 'Size');

model.result.table('tbl1').comments('Line Average 1');
model.result.table('tbl2').comments('Point Evaluation 1');

model.component('comp1').view('view1').set('renderwireframe', true);
model.component('comp1').view('view1').set('showgrid', false);
model.component('comp1').view('view1').set('scenelight', false);
model.component('comp1').view('view1').set('transparency', true);
model.component('comp1').view('view2').axis.set('xmin', 73.66307067871094);
model.component('comp1').view('view2').axis.set('xmax', 89.07933807373047);
model.component('comp1').view('view2').axis.set('ymin', 22.896543502807617);
model.component('comp1').view('view2').axis.set('ymax', 30.913537979125977);
model.component('comp1').view('view3').label('View 3.1');
model.component('comp1').view('view3').axis.set('xmin', -639.5604248046875);
model.component('comp1').view('view3').axis.set('xmax', 1439.5604248046875);
model.component('comp1').view('view3').axis.set('ymin', -555.1805419921875);
model.component('comp1').view('view3').axis.set('ymax', 555.1805419921875);
model.component('comp1').view('view6').label('View 6');
model.component('comp1').view('view6').axis.set('xmin', -6.906982421875);
model.component('comp1').view('view6').axis.set('xmax', 214.30697631835938);
model.component('comp1').view('view6').axis.set('ymin', -59.070003509521484);
model.component('comp1').view('view6').axis.set('ymax', 59.070003509521484);
model.component('comp1').view('view7').label('View 7');
model.component('comp1').view('view7').axis.set('xmin', 48.396507263183594);
model.component('comp1').view('view7').axis.set('xmax', 159.00347900390625);
model.component('comp1').view('view7').axis.set('ymin', -2.684999465942383);
model.component('comp1').view('view7').axis.set('ymax', 56.38500213623047);

model.component('comp1').material('mat2').label('Air');
model.component('comp1').material('mat2').set('family', 'air');
model.component('comp1').material('mat2').propertyGroup('def').func('eta').set('arg', 'T');
model.component('comp1').material('mat2').propertyGroup('def').func('eta').set('pieces', {'200.0' '1600.0' '-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4'});
model.component('comp1').material('mat2').propertyGroup('def').func('Cp').set('arg', 'T');
model.component('comp1').material('mat2').propertyGroup('def').func('Cp').set('pieces', {'200.0' '1600.0' '1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4'});
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('expr', 'pA*0.02897/R_const[K*mol/J]/T');
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('args', {'pA' 'T'});
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('dermethod', 'manual');
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('argders', {'pA' 'd(pA*0.02897/R_const/T,pA)'; 'T' 'd(pA*0.02897/R_const/T,T)'});
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('argunit', {'' ''});
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('plotaxis', {'on' 'on'});
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('plotfixedvalue', {'0' '0'});
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('plotargs', {'pA' '0' '1'; 'T' '0' '1'});
model.component('comp1').material('mat2').propertyGroup('def').func('k').set('arg', 'T');
model.component('comp1').material('mat2').propertyGroup('def').func('k').set('pieces', {'200.0' '1600.0' '-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4'});
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('expr', 'sqrt(1.4*287*T)');
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('args', {'T'});
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('dermethod', 'manual');
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('argders', {'T' 'd(sqrt(1.4*287*T),T)'});
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('argunit', {''});
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('plotargs', {'T' '0' '1'});
model.component('comp1').material('mat2').propertyGroup('def').set('relpermeability', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat2').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat2').propertyGroup('def').set('dynamicviscosity', 'eta(T[1/K])[Pa*s]');
model.component('comp1').material('mat2').propertyGroup('def').set('ratioofspecificheat', '1.4');
model.component('comp1').material('mat2').propertyGroup('def').set('electricconductivity', {'1[S/m]' '0' '0' '0' '1[S/m]' '0' '0' '0' '1[S/m]'});
model.component('comp1').material('mat2').propertyGroup('def').set('heatcapacity', 'Cp(T[1/K])[J/(kg*K)]');
model.component('comp1').material('mat2').propertyGroup('def').set('density', 'rho(pA[1/Pa],T[1/K])[kg/m^3]');
model.component('comp1').material('mat2').propertyGroup('def').set('thermalconductivity', {'k(T[1/K])[W/(m*K)]' '0' '0' '0' 'k(T[1/K])[W/(m*K)]' '0' '0' '0' 'k(T[1/K])[W/(m*K)]'});
model.component('comp1').material('mat2').propertyGroup('def').set('soundspeed', 'cs(T[1/K])[m/s]');
model.component('comp1').material('mat2').propertyGroup('def').addInput('temperature');
model.component('comp1').material('mat2').propertyGroup('def').addInput('pressure');
model.component('comp1').material('mat2').propertyGroup('RefractiveIndex').set('n', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});

model.component('comp1').cpl('intop1').active(false);

model.common('cminpt').label('Common model inputs 1');

model.component('comp1').physics('mf').feature('al1').set('minput_temperature_src', 'userdef');
model.component('comp1').physics('mf').feature('al1').set('AllowDeprecatedCurves', true);
model.component('comp1').physics('mf').feature('al1').set('normBr_crel_BH_RemanentFluxDensity_mat', 'userdef');
model.component('comp1').physics('mf').feature('al1').set('AllowSolidMaterials', true);
model.component('comp1').physics('mf').feature('al1').label(['Amp' native2unicode(hex2dec({'00' 'e8'}), 'unicode') 're''s Law 1']);
model.component('comp1').physics('mf').feature('al1').featureInfo('information').label('Migrated Feature');
model.component('comp1').physics('mf').feature('dcont1').set('pairDisconnect', true);
model.component('comp1').physics('mf').feature('dcont1').set('pairSelection', 'all');
model.component('comp1').physics('mf').feature('dcont1').active(false);
model.component('comp1').physics('mf').feature('dcont1').label('Continuity');
model.component('comp1').physics('mf').feature('coil1').set('ConductorModel', 'Multi');
model.component('comp1').physics('mf').feature('coil1').set('ICoil', 1);
model.component('comp1').physics('mf').feature('coil1').set('N', 'Nt*Nc/2');
model.component('comp1').physics('mf').feature('coil1').set('sigmaCoil', '0[S/m]');
model.component('comp1').physics('mf').feature('coil1').set('AreaFrom', 'UserDefined');
model.component('comp1').physics('mf').feature('coil1').set('coilWindArea', 'thk*wid');
model.component('comp1').physics('mf').feature('coil1').set('minput_temperature_src', 'userdef');
model.component('comp1').physics('mf').feature('coil1').set('AllowDeprecatedCurves', true);
model.component('comp1').physics('mf').feature('coil1').set('normBr_crel_BH_RemanentFluxDensity_mat', 'userdef');
model.component('comp1').physics('mf').feature('coil1').set('AllowAllMaterialModelsInCoil', true);
model.component('comp1').physics('mf').feature('coil1').set('HarmonicLoss', false);
model.component('comp1').physics('mf').feature('coil1').label('Coil 1');
model.component('comp1').physics('mf').feature('coil1').featureInfo('information').label('Migrated Feature');
model.component('comp1').physics('mf').feature('symp1').set('Symmetry_type', 'Antisymmetry');
model.component('comp1').physics('mf').feature('symp1').active(false);

model.component('comp1').mesh('mesh1').feature('ftri1').active(false);
model.component('comp1').mesh('mesh1').feature('ftri1').set('smoothcontrol', true);
model.component('comp1').mesh('mesh1').feature('ftri1').feature('size1').set('hauto', 1);
model.component('comp1').mesh('mesh1').feature('ftri1').feature('size1').set('custom', 'on');
model.component('comp1').mesh('mesh1').feature('ftri1').feature('size1').set('hmax', 0.15);
model.component('comp1').mesh('mesh1').feature('ftri1').feature('size1').set('hmaxactive', true);
model.component('comp1').mesh('mesh1').feature('ftri1').feature('size1').set('hmin', 0.1);
model.component('comp1').mesh('mesh1').feature('ftri1').feature('size1').set('hminactive', false);
model.component('comp1').mesh('mesh1').feature('swe1').active(false);
model.component('comp1').mesh('mesh1').feature('swe1').set('element', 'hexlegacy54');
model.component('comp1').mesh('mesh1').feature('swe1').set('smoothcontrol', true);
model.component('comp1').mesh('mesh1').feature('swe1').feature('dis1').set('numelem', 8);
model.component('comp1').mesh('mesh1').feature('conv1').active(false);
model.component('comp1').mesh('mesh1').feature('ftet1').set('smoothcontrol', true);
model.component('comp1').mesh('mesh1').feature('ftet1').set('optcurved', false);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hauto', 1);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('custom', 'on');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmax', '0.08[m]');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmaxactive', true);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmin', '0.01[m]');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hminactive', true);
model.component('comp1').mesh('mesh1').feature('ftet2').set('smoothcontrol', true);
model.component('comp1').mesh('mesh1').feature('ftet2').set('optcurved', false);
model.component('comp1').mesh('mesh1').feature('ftet2').feature('size1').set('hauto', 1);
model.component('comp1').mesh('mesh1').feature('ftet2').feature('size1').set('custom', 'on');
model.component('comp1').mesh('mesh1').feature('ftet2').feature('size1').set('hmax', '0.5[m]');
model.component('comp1').mesh('mesh1').feature('ftet2').feature('size1').set('hmaxactive', true);
model.component('comp1').mesh('mesh1').feature('ftet2').feature('size1').set('hmin', '0.01[m]');
model.component('comp1').mesh('mesh1').feature('ftet2').feature('size1').set('hminactive', true);
model.component('comp1').mesh('mesh1').feature('ftet3').set('smoothcontrol', true);
model.component('comp1').mesh('mesh1').feature('ftet3').set('optcurved', false);
model.component('comp1').mesh('mesh1').feature('ftet3').feature('size1').set('hauto', 1);
model.component('comp1').mesh('mesh1').feature('ftet3').feature('size1').set('custom', 'on');
model.component('comp1').mesh('mesh1').feature('ftet3').feature('size1').set('hmax', '0.5[m]');
model.component('comp1').mesh('mesh1').feature('ftet3').feature('size1').set('hmaxactive', true);
model.component('comp1').mesh('mesh1').feature('ftet3').feature('size1').set('hmin', '0.01[m]');
model.component('comp1').mesh('mesh1').feature('ftet3').feature('size1').set('hminactive', true);
model.component('comp1').mesh('mesh1').run;

model.study.create('std1');
model.study('std1').create('ccc', 'CoilCurrentCalculation');
model.study('std1').create('stat', 'Stationary');

model.sol.create('sol1');
model.sol('sol1').attach('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').create('su1', 'StoreSolution');
model.sol('sol1').create('st2', 'StudyStep');
model.sol('sol1').create('v2', 'Variables');
model.sol('sol1').create('s2', 'Stationary');
model.sol('sol1').feature('s1').create('se1', 'Segregated');
model.sol('sol1').feature('s1').feature('se1').create('ss1', 'SegregatedStep');
model.sol('sol1').feature('s1').feature('se1').create('ss2', 'SegregatedStep');
model.sol('sol1').feature('s1').feature('se1').feature.remove('ssDef');
model.sol('sol1').feature('s1').feature.remove('fcDef');
model.sol('sol1').feature('s2').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s2').create('i1', 'Iterative');
model.sol('sol1').feature('s2').feature('i1').create('asamg1', 'AuxiliarySpaceAMG');
model.sol('sol1').feature('s2').feature.remove('fcDef');

model.result.dataset.create('cpt1', 'CutPoint3D');
model.result.dataset.create('cln1', 'CutLine3D');
model.result.dataset.create('cln2', 'CutLine3D');
model.result.dataset.create('mir1', 'Mirror3D');
model.result.dataset.create('mir2', 'Mirror3D');
model.result.dataset.create('int4_ds1', 'Grid1D');
model.result.dataset.create('ptds1', 'CutPoint1D');
model.result.dataset.create('int5_ds1', 'Grid1D');
model.result.dataset.create('ptds2', 'CutPoint1D');
model.result.dataset.create('pc1', 'ParCurve3D');
model.result.dataset('dset1').selection.geom('geom1', 3);
model.result.dataset('dset1').selection.all;
model.result.dataset('int4_ds1').set('data', 'none');
model.result.dataset('ptds1').set('data', 'int4_ds1');
model.result.dataset('int5_ds1').set('data', 'none');
model.result.dataset('ptds2').set('data', 'int5_ds1');
model.result.numerical.create('int1', 'IntLine');
model.result.numerical.create('gev1', 'EvalGlobal');
model.result.numerical.create('int2', 'IntSurface');
model.result.numerical.create('av1', 'AvLine');
model.result.numerical.create('pev1', 'EvalPoint');
model.result.create('pg4', 'PlotGroup3D');
model.result.create('pg7', 'PlotGroup3D');
model.result.create('pg8', 'PlotGroup3D');
model.result.create('pg10', 'PlotGroup3D');
model.result.create('pgJc', 'PlotGroup3D');
model.result.create('pgLF', 'PlotGroup3D');
model.result('pg4').selection.geom('geom1', 3);
model.result('pg4').selection.set([2]);
model.result('pg4').create('surf1', 'Surface');
model.result('pg4').create('vol1', 'Volume');
model.result('pg4').feature('surf1').set('expr', 'mf.normB*420');
model.result('pg4').feature('vol1').set('expr', 'mf.normB*Ip');
model.result('pg7').selection.geom('geom1', 3);
model.result('pg7').selection.set([2]);
model.result('pg7').create('surf1', 'Surface');
model.result('pg7').create('vol1', 'Volume');
model.result('pg7').feature('surf1').set('expr', '420/IcB20(420*abs(Br))');
model.result('pg7').feature('vol1').set('expr', '444/(IcB20(444*abs(Br))*wid/10[mm])');
model.result('pg8').selection.geom('geom1', 3);
model.result('pg8').selection.set([2]);
model.result('pg8').create('surf1', 'Surface');
model.result('pg8').feature('surf1').set('expr', 'Ip*abs(Br)');
model.result('pg10').selection.geom('geom1', 3);
model.result('pg10').selection.set([3]);
model.result('pg10').create('vol1', 'Volume');
model.result('pg10').feature('vol1').set('expr', 'Ip*mf.normB');
model.result('pgJc').selection.geom('geom1', 3);
model.result('pgJc').selection.set([2]);
model.result('pgJc').create('v', 'Volume');
model.result('pgJc').feature('v').set('expr', 'Jc_mang');
model.result('pgLF').selection.geom('geom1', 3);
model.result('pgLF').selection.set([2]);
model.result('pgLF').create('v', 'Volume');
model.result('pgLF').feature('v').set('expr', 'Ip/Jc_mang*wid/10[mm]');

model.sol('sol1').feature('st1').label('Compile Equations: Coil Geometry Analysis');
model.sol('sol1').feature('v1').label('Dependent Variables 1.1');
model.sol('sol1').feature('s1').label('Stationary Solver 1.1');
model.sol('sol1').feature('s1').set('probesel', 'none');
model.sol('sol1').feature('s1').feature('dDef').label('Direct 1');
model.sol('sol1').feature('s1').feature('aDef').label('Advanced 1');
model.sol('sol1').feature('s1').feature('se1').label('Segregated 1.1');
model.sol('sol1').feature('s1').feature('se1').set('segterm', 'itertol');
model.sol('sol1').feature('s1').feature('se1').set('segiter', 6);
model.sol('sol1').feature('s1').feature('se1').feature('ss1').label('Segregated Step 1.1');
model.sol('sol1').feature('s1').feature('se1').feature('ss1').set('segvar', {'comp1_mf_coil1_ccc1_s'});
model.sol('sol1').feature('s1').feature('se1').feature('ss2').label('Segregated Step 2.1');
model.sol('sol1').feature('s1').feature('se1').feature('ss2').set('segvar', {'comp1_mf_coil1_ccc1_p' 'comp1_mf_coil1_ccc1_lm'});
model.sol('sol1').feature('su1').label('Solution Store 1.1');
model.sol('sol1').feature('st2').label('Compile Equations: Stationary');
model.sol('sol1').feature('st2').set('studystep', 'stat');
model.sol('sol1').feature('v2').label('Dependent Variables 2.1');
model.sol('sol1').feature('v2').set('initmethod', 'sol');
model.sol('sol1').feature('v2').set('initsol', 'sol1');
model.sol('sol1').feature('v2').set('solnum', 'auto');
model.sol('sol1').feature('v2').set('notsolmethod', 'sol');
model.sol('sol1').feature('v2').set('notsol', 'sol1');
model.sol('sol1').feature('v2').set('notsolnum', 'auto');
model.sol('sol1').feature('s2').label('Stationary Solver 2.1');
model.sol('sol1').feature('s2').feature('dDef').label('Direct 1');
model.sol('sol1').feature('s2').feature('aDef').label('Advanced 1');
model.sol('sol1').feature('s2').feature('fc1').label('Fully Coupled 1.1');
model.sol('sol1').feature('s2').feature('i1').label('Iterative 1.1');
model.sol('sol1').feature('s2').feature('i1').set('linsolver', 'fgmres');
model.sol('sol1').feature('s2').feature('i1').set('nlinnormuse', true);
model.sol('sol1').feature('s2').feature('i1').feature('ilDef').label('Incomplete LU 1');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').label('Auxiliary-Space AMG 1.1');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').set('prelowerorder', 'sor');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').set('postlowerorder', 'sor');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').feature('pr').label('AMG Presmoother 1');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').feature('pr').feature('soDef').label('SOR 1');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').feature('po').label('AMG Postsmoother 1');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').feature('po').feature('soDef').label('SOR 1');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').feature('cs').label('AMG Coarse Solver 1');
model.sol('sol1').feature('s2').feature('i1').feature('asamg1').feature('cs').feature('dDef').label('Direct 1');
model.sol('sol2').label('Solution Store 1');

model.study('std1').runNoGen;

model.result.dataset('cpt1').set('pointx', 'R0');
model.result.dataset('cpt1').set('pointy', 0);
model.result.dataset('cpt1').set('pointz', 0);
model.result.dataset('cln1').set('genpoints', {'8.5' '0' '0'; '8.5' '0' 'wid/2'});
model.result.dataset('cln2').set('genpoints', {'59.6' '0' '0'; '59.6' '0' 'wid/2'});
model.result.dataset('cln2').set('spacevars', {'cln1x'});
model.result.dataset('mir1').set('quickplane', 'xy');
model.result.dataset('mir2').set('quickplane', 'zx');
model.result.dataset('int4_ds1').set('function', 'int4');
model.result.dataset('int4_ds1').set('par1', 't');
model.result.dataset('int4_ds1').set('parmin1', -0.2);
model.result.dataset('int4_ds1').set('parmax1', 2.2);
model.result.dataset('int4_ds1').set('res1', 10000);
model.result.dataset('ptds1').set('pointx', '0.0 1.0 2.0 ');
model.result.dataset('int5_ds1').set('function', 'int5');
model.result.dataset('int5_ds1').set('par1', 't');
model.result.dataset('int5_ds1').set('parmin1', -0.30000000000000004);
model.result.dataset('int5_ds1').set('parmax1', 3.3);
model.result.dataset('int5_ds1').set('res1', 10000);
model.result.dataset('ptds2').set('pointx', '0.0 1.0 2.0 3.0 ');
model.result.dataset('pc1').label('Parameterized Curve 3D 1');
model.result.dataset('pc1').set('exprx', 'R0*cos(2*pi*s)');
model.result.dataset('pc1').set('exprz', 'R0*sin(2*pi*s)');
model.result.numerical('int1').label(' HTS tape of one TF coil');
model.result.numerical('int1').set('expr', {'1*Nt*Nc*2'});
model.result.numerical('int1').set('unit', {'cm'});
model.result.numerical('int1').set('descr', {''});
model.result.numerical('gev1').set('expr', {'intop1(1)'});
model.result.numerical('gev1').set('unit', {''});
model.result.numerical('gev1').set('descr', {'Integration 1'});
model.result.numerical('int2').set('expr', {'1'});
model.result.numerical('int2').set('unit', {'m^2'});
model.result.numerical('int2').set('descr', {''});
model.result.numerical('av1').set('data', 'cln2');
model.result.numerical('av1').set('table', 'tbl1');
model.result.numerical('av1').set('expr', {'mf.Bx'});
model.result.numerical('av1').set('unit', {'T'});
model.result.numerical('av1').set('descr', {'Magnetic flux density, x component'});
model.result.numerical('pev1').set('data', 'cpt1');
model.result.numerical('pev1').set('table', 'tbl2');
model.result.numerical('pev1').set('expr', {'-mf.Bz*420'});
model.result.numerical('pev1').set('unit', {'T'});
model.result.numerical('pev1').set('descr', {''});
model.result.numerical('av1').setResult;
model.result.numerical('pev1').setResult;
model.result('pg4').label('Bnorm');
model.result('pg4').set('showlegendsmaxmin', true);
model.result('pg4').set('smooth', 'internal');
model.result('pg4').feature('surf1').active(false);
model.result('pg4').feature('surf1').set('resolution', 'normal');
model.result('pg4').feature('vol1').set('resolution', 'normal');
model.result('pg7').label('Iop//Ic');
model.result('pg7').set('showlegendsmaxmin', true);
model.result('pg7').set('smooth', 'internal');
model.result('pg7').feature('surf1').active(false);
model.result('pg7').feature('surf1').set('resolution', 'normal');
model.result('pg7').feature('vol1').set('resolution', 'normal');
model.result('pg8').label('Br');
model.result('pg8').set('showlegendsmaxmin', true);
model.result('pg8').set('smooth', 'internal');
model.result('pg8').feature('surf1').set('resolution', 'normal');
model.result('pg10').label('Bcen');
model.result('pg10').set('showlegendsmaxmin', true);
model.result('pg10').set('smooth', 'internal');
model.result('pg10').feature('vol1').set('resolution', 'normal');
model.result('pgJc').label('Jc_mang[A/mm2]');
model.result('pgJc').set('showlegendsmaxmin', true);
model.result('pgJc').feature('v').set('resolution', 'normal');
model.result('pgLF').label('load factor Mangiarotti');
model.result('pgLF').set('showlegendsmaxmin', true);
model.result('pgLF').feature('v').set('resolution', 'normal');

out = model;
