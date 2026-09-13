function ipT_resolve_wid4()
  % arc_wid_4mm 变体: 带材宽度 4mm (vs ARC 12mm), Nc=12 不变。
  % wid 改变两处: (1) Heig=(wid+dist)*Nc -> 更紧凑WP -> 峰值场↑(物理真实,非bug);
  %               (2) Ic = Jc_mang*wid_cm -> 带材面积小 -> Ic↓。两者都使 Ip↓。
  % 安匝标定: amp_turns=Nt_fem*Nc*Ip=8.4MA, wid 不入安匝 -> Ip_cal=8.4e6/(1750*12)=400 (同ARC)。
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'ipT_wid4.log']);
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    Nc=12; AT=8.4e6; lf=0.7; wid_mm=4; wid_cm=wid_mm/10;
    Nt_fem = str2double(char(m.param.get('Nt')));
    Ip_cal = AT/(Nt_fem*Nc);
    m.param.set('wid', [num2str(wid_mm) '[mm]']);   % 4mm -> Heig=(4+3)*12=84mm (ARC 180mm)
    m.param.set('Ip', num2str(Ip_cal));
    m.geom('geom1').run(); m.mesh('mesh1').run();
    fprintf('wid=%dmm Nc=%d Nt_fem=%g Ip_cal=%.1f amp_turns=%.3e (Heig=%.0fmm)\n', ...
            wid_mm,Nc,Nt_fem,Ip_cal,Nt_fem*Nc*Ip_cal,(wid_mm+3)*Nc);
    fprintf(' T[K]  maxB[T]  minJc[A/mm2 or A/cm?]  Ic(=Jc*%.1f)  Ip(0.7)  Ip取整  Nt_design\n', wid_cm);
    M=[];
    for T=[4.2 10 20]
      m.param.set('Top', num2str(T));
      m.sol('sol1').runAll;
      maxB  = mphmax(m,'Bmag_T','volume','selection',2,'dataset','dset1');
      Jcmin = mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
      Ic=Jcmin*wid_cm; Ip=lf*Ic; Ip_r=round(Ip/5)*5; Nt=AT/(Nc*Ip_r);
      fprintf(' %4.1f  %6.1f  %12.0f  %9.1f  %7.1f  %5d  %7.0f\n', T,maxB,Jcmin,Ic,Ip,Ip_r,Nt);
      M=[M; T maxB Jcmin Ic Ip Ip_r Nt];
    end
    writematrix(M,[res 'ip_vs_T_wid4.csv']);
    m.param.set('Top','20'); m.sol('sol1').runAll;
    mphsave(m,[base 'ARC_field_mangiarotti_wid4.mph']);
    fprintf('WID4_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
