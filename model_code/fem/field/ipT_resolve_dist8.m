function ipT_resolve_dist8()
  % ARC 变体: 饼间距 dist 3->8mm (vs ARC 3mm), Nc=12/WID=12mm 不变。
  % Heig=(12+8)*12=240mm (=Nc16 同包络, 应快且不卡; dist=10mm的264mm会卡网格)。
  % H_magnet=(WID+DIST)*NP=240mm -> V_magnet x1.33 -> 核热x1.33 (device.py 自动算)。
  % 安匝标定: Ip_cal=8.4e6/(1750*12)=400。Ic=Jc*wid_cm=Jc*1.2。先解20K存盘, 再补4.2/10K。
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'ipT_dist8.log']);
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    Nc=12; AT=8.4e6; lf=0.7; wid_cm=1.2; dist_mm=8;
    Nt_fem = str2double(char(m.param.get('Nt')));
    Ip_cal = AT/(Nt_fem*Nc);
    m.param.set('dist', [num2str(dist_mm) '[mm]']);
    m.param.set('Ip', num2str(Ip_cal));
    fprintf('dist=8mm rebuilding geom+mesh (Heig=%dmm)...\n',(12+dist_mm)*Nc);
    m.geom('geom1').run(); m.mesh('mesh1').run();
    % 先解 20K + 存盘 (供GUI查看)
    m.param.set('Top','20'); m.sol('sol1').runAll;
    maxB=mphmax(m,'Bmag_T','volume','selection',2,'dataset','dset1');
    Jcmin=mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
    Ic=Jcmin*wid_cm; Ip=lf*Ic; Ip_r=round(Ip/10)*10; Nt=AT/(Nc*Ip_r);
    fprintf('20K: maxB=%.1fT minJc=%.0f Ip=%.0f Nt=%.0f\n', maxB,Jcmin,Ip_r,Nt);
    mphsave(m,[base 'ARC_field_mangiarotti_dist8.mph']);
    fprintf('DIST8_SAVED -> ARC_field_mangiarotti_dist8.mph\n');
    M=[20 maxB Jcmin Ic Ip Ip_r Nt];
    for T=[4.2 10]
      m.param.set('Top', num2str(T)); m.sol('sol1').runAll;
      b=mphmax(m,'Bmag_T','volume','selection',2,'dataset','dset1');
      jc=mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
      ic=jc*wid_cm; ip=lf*ic; ipr=round(ip/10)*10; nt=AT/(Nc*ipr);
      fprintf('%.1fK: maxB=%.1fT minJc=%.0f Ip=%.0f Nt=%.0f\n', T,b,jc,ipr,nt);
      M=[M; T b jc ic ip ipr nt];
    end
    writematrix(M,[res 'ip_vs_T_dist8.csv']);
    fprintf('DIST8_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
