function ipT_sweep_wid4_Nc()
  % wid=4mm: 扫描饼数 Nc, 找使径向可行 (Nt_req<=Nt_max) 的最小饼数。
  % 物理: 饼多 -> 磁体高 -> 电流分散 -> 峰值场降 -> Jc升 -> Ip回升; 同时 Nt_req=8.4e6/(Nc*Ip) 降。
  % 约束: Nt_req <= Nt_max = R2/thk = 0.64/0.32mm = 2000 匝/饼。
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'ipT_sweep_wid4_Nc.log']);
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    AT=8.4e6; lf=0.7; wid_mm=4; wid_cm=0.4; thk=0.32e-3; R2=0.64;
    Nt_max = R2/thk;
    Nt_fem = str2double(char(m.param.get('Nt')));
    m.param.set('wid', [num2str(wid_mm) '[mm]']);
    fprintf('wid=4mm, Nt_max(径向可容)=%.0f 匝/饼\n', Nt_max);
    fprintf('%-5s %-8s %-7s | %-22s | %-22s | %-22s\n','Nc','Heig','maxB', ...
            'T=4.2: Ip / Nt_req(可行?)','T=10: Ip / Nt_req','T=20: Ip / Nt_req');
    M=[];
    for Nc=[24 32 40 48]
      Ip_cal = AT/(Nt_fem*Nc);
      m.param.set('Nc', num2str(Nc));
      m.param.set('Ip', num2str(Ip_cal));
      m.geom('geom1').run(); m.mesh('mesh1').run();
      Heig=(wid_mm+3)*Nc;
      row=sprintf('%-5d %-8.0f ', Nc, Heig);
      maxB=NaN; cells={};
      for T=[4.2 10 20]
        m.param.set('Top', num2str(T)); m.sol('sol1').runAll;
        if isnan(maxB), maxB=mphmax(m,'Bmag_T','volume','selection',2,'dataset','dset1'); end
        Jcmin=mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
        Ic=Jcmin*wid_cm; Ip=lf*Ic; Ntreq=AT/(Nc*Ip);
        ok=''; if Ntreq<=Nt_max, ok='OK'; else ok='X'; end
        cells{end+1}=sprintf('%.0f / %.0f(%s)', Ip, Ntreq, ok);
        M=[M; Nc T Heig maxB Ip Ntreq];
      end
      fprintf('%s%-7.1f | %-22s | %-22s | %-22s\n', row, maxB, cells{1},cells{2},cells{3});
    end
    writematrix(M,[res 'sweep_wid4_Nc.csv']);
    fprintf('SWEEP_WID4_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
